import cv2
import numpy as np
from skimage.measure import label, regionprops
from scipy.optimize import linear_sum_assignment
from KalFilter import KalmanFilter


class MotionDetector:
    def __init__(self, alpha=5, tau=25, skip=1, max_objects=10, delta=50):
        self.alpha = alpha  # object lifetime
        self.tau = tau      # threshold for motion detection
        self.skip = skip    # process every nth frame
        self.max_objects = max_objects
        self.delta = delta  # max distance for matching
        self.objects = []
        self.frames = []
        print(
            f"alpha={self.alpha}, tau={self.tau}, skip={self.skip}, max_objects={self.max_objects}, delta={self.delta}")

    def motion_blob_detection(self, ft, ft1, ft2):
        """3-frame motion detection"""
        # Convert to grayscale
        ft = cv2.cvtColor(ft, cv2.COLOR_RGB2GRAY)
        ft1 = cv2.cvtColor(ft1, cv2.COLOR_RGB2GRAY)
        ft2 = cv2.cvtColor(ft2, cv2.COLOR_RGB2GRAY)

        # Find motion by taking minimum of frame differences
        diff1 = cv2.absdiff(ft1, ft2)
        diff2 = cv2.absdiff(ft, ft1)
        motion = cv2.min(diff1, diff2)

        # Threshold and dilate to get clean blobs
        _, motion = cv2.threshold(
            motion, thresh=self.tau, maxval=255, type=cv2.THRESH_BINARY)
        motion = cv2.dilate(motion, kernel=np.ones((9, 9), np.uint8))
        return motion

    def extract_detections(self, motion):
        """Extract centroids and bounding boxes from motion blobs
        get x, y coordinate, return centroids and bounding boxes properties
        centroid is (y, x) in skimage, bbox is (min_row, min_col, max_row, max_col)
        return centroids as (x, y) and bounding boxes
        """
        labeled = label(motion)
        props = regionprops(labeled)

        return [((prop.centroid[1], prop.centroid[0]), prop.bbox)
                for prop in props if prop.area >= 50]

    def reset(self):
        self.frames = []
        self.objects = []

    def draw_bounding_boxes(self, frame, objects):
        """Draw bounding boxes and history trails for tracked objects"""
        frame_with_boxes = frame.copy()

        for obj in objects:
            self._draw_car(frame_with_boxes, obj)

        return frame_with_boxes

    def _draw_car(self, frame, obj):
        """Draw bounding box and trail for a single object"""
        # Draw bounding box
        self._draw_bbox(frame, obj.bbox)

        # Draw history trail
        if hasattr(obj, 'history') and len(obj.history) > 1:
            self._draw_line(frame, obj.history)

    def _draw_bbox(self, frame, bbox):
        """Draw a single bounding box"""
        # bbox format: (min_row, min_col, max_row, max_col)
        # cv2.rectangle expects: (x1, y1), (x2, y2) where x=col, y=row
        top_left = (bbox[1], bbox[0])      # (min_col, min_row)
        bottom_right = (bbox[3], bbox[2])  # (max_col, max_row)

        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), thickness=1)

    def _draw_line(self, frame, history):
        """Draw the movement trail for an object"""
        for i in range(1, len(history)):
            # Get previous and current points
            pt1 = self._point_to_int(history[i - 1])  # (x-1, y-1)
            pt2 = self._point_to_int(history[i])  # (x, y)

            cv2.line(frame, pt1, pt2, (0, 255, 0), thickness=2)

    def _point_to_int(self, point):
        """Convert point coordinates to integers for OpenCV
        point[0] = x coordinate (column)
        point[1] = y coordinate (row)
        """
        return (int(round(point[0])), int(round(point[1])))

    def add_frame(self, frame):
        if len(frame.shape) != 3:
            raise ValueError("Frames must be a 3D numpy array.")
        self.frames.append(frame)
        if len(self.frames) > 3:
            self.frames.pop(0)

    def update(self, new_frame, frame_idx=None):
        """Update tracking with new frame"""
        # Skip processing if not at skip interval
        if frame_idx is not None and frame_idx % self.skip != 0:
            return self.draw_bounding_boxes(new_frame, self.objects)

        # Add frame and check if we can process
        self.add_frame(new_frame)
        if len(self.frames) < 3:
            return self.draw_bounding_boxes(new_frame, self.objects)

        # Process tracking
        return self._process_tracking(new_frame)

    def _process_tracking(self, new_frame):
        """Main tracking logic"""
        # Get motion detections
        ft, ft1, ft2 = self.frames[-3], self.frames[-2], self.frames[-1]
        motion = self.motion_blob_detection(ft, ft1, ft2)
        detections = self.extract_detections(motion)

        # Update existing objects and create new ones
        self._update_objects_with_detections(detections)

        return self.draw_bounding_boxes(new_frame, self.objects)

    def _update_objects_with_detections(self, detections):
        """Update existing objects and create new ones from detections
        there will a 1 frame delay in object creation, since we need to have 3 frames first
        """
        # Predict where existing objects should be
        for obj in self.objects:
            obj.predict()

        # Match detections to existing objects
        used_detections = set()
        for obj in self.objects:
            best_detection = self._find_closest_detection(
                obj, detections, used_detections)
            if best_detection is not None:
                det_idx, (centroid, bbox) = best_detection
                obj.update(np.array(centroid).reshape(2, 1))
                obj.bbox = bbox
                obj.life = self.alpha
                obj.history.append(centroid)
                if len(obj.history) > 10:  # record up to 10 frames history
                    obj.history.pop(0)
                used_detections.add(det_idx)
            else:
                obj.life -= 1

        # Remove dead objects
        self.objects = [obj for obj in self.objects if obj.life > 0]

        # Create new objects from unused detections
        for i, (centroid, bbox) in enumerate(detections):
            if i not in used_detections and len(self.objects) < self.max_objects:
                kf = KalmanFilter.create_kalman_filter(
                    position=centroid, life=self.alpha)
                kf.bbox = bbox
                kf.history = [centroid]
                self.objects.append(kf)

    def _find_closest_detection(self, obj, detections, used_detections):
        """Find the closest detection to an object using regular euclidean distance"""
        best_match = None
        best_dist = float('inf')

        # (x, y) position
        obj_pos = np.array([obj.state[0, 0], obj.state[1, 0]])
        # for each detection, find the one closest to the object's predicted position
        # detections are in the form of (centroid, bbox)
        for i, (centroid, bbox) in enumerate(detections):
            if i in used_detections:
                continue

            dist = np.linalg.norm(np.array(centroid) - obj_pos)
            if dist < self.delta and dist < best_dist:
                best_dist = dist
                best_match = (i, (centroid, bbox))

        # the detection index and its details (centroid, bbox)
        return best_match
