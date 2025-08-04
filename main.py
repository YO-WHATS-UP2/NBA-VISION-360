import os
import cv2
import pytesseract
from utils.video_utils import read_video, save_video 
from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker
from drawers.player_tracks_drawer import PlayerTracksDrawer
from drawers.ball_tracks_drawer import BallTracksDrawer
from team_assigner.team_assigner import TeamAssigner
from ball_aquisition.ball_aquisition_detector import BallAquisitionDetector
from drawers.team_ball_control_drawer import TeamBallControlDrawer
from pass_and_interception_detector.pass_and_interception_detector import PassAndInterceptionDetector
from drawers.pass_and_interceptions_drawer import PassInterceptionDrawer
from court_keypoint_detector.court_keypoint_detector import CourtKeypointDetector
from drawers.court_keypoints_drawer import CourtKeypointDrawer
from drawers.tactical_view_drawer import TacticalViewDrawer
from tactical_view_convertor.tactical_view_converter import TacticalViewConverter
from speed_and_distance_calculator.speed_and_distance_calculator import SpeedAndDistanceCalculator
from drawers.frame_number_drawer import FrameNumberDrawer
from drawers.speed_and_distance_drawer import SpeedAndDistanceDrawer
from drawers.player_heatmap_generator import PlayerHeatmapGenerator
from player_name_mapper import PlayerNameMapper
from predictor import overlay_win_probability_on_frames

from configs.configs import (
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    COURT_KEYPOINT_DETECTOR_PATH,
    OUTPUT_VIDEO_PATH
)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def main():
    video_frames = read_video("input_videos/video_1.mp4")

    # Trackers and detectors
    player_tracker = PlayerTracker(PLAYER_DETECTOR_PATH)
    ball_tracker = BallTracker(BALL_DETECTOR_PATH)
    court_keypoint_detector = CourtKeypointDetector(COURT_KEYPOINT_DETECTOR_PATH)

    player_tracks = player_tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path="stubs/player_tracks_stubs.pkl")
    ball_tracks = ball_tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path="stubs/ball_tracks_stubs.pkl")
    court_keypoints = court_keypoint_detector.get_court_keypoints(video_frames, read_from_stub=False, stub_path="stubs/court_keypoints_stubs.pkl")

    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(video_frames, player_tracks, read_from_stub=False, stub_path="stubs/player_assignment_stubs.pkl")

    ball_aquisition_detector = BallAquisitionDetector()
    ball_aquisition = ball_aquisition_detector.detect_ball_possession(player_tracks, ball_tracks)

    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(ball_aquisition, player_assignment)
    interceptions = pass_and_interception_detector.detect_interceptions(ball_aquisition, player_assignment)

    tactical_view_converter = TacticalViewConverter("images/basketball_court.png")
    court_keypoints_per_frame = tactical_view_converter.validate_keypoints(court_keypoints)
    tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(court_keypoints_per_frame, player_tracks)

    speed_and_distance_calculator = SpeedAndDistanceCalculator(
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.actual_width_in_meters,
        tactical_view_converter.actual_height_in_meters
    )
    player_distances_per_frame = speed_and_distance_calculator.calculate_distance(tactical_player_positions)
    player_speed_per_frame = speed_and_distance_calculator.calculate_speed(player_distances_per_frame)

    # Optional: player name mapping
    player_mapper = PlayerNameMapper("D:/basketball ml - Copy - Copy/real-player-data.basketball.json", "D:/basketball ml - Copy - Copy/real-player-stats.basketball.json")
    player_mapper.assign_player_to_yolo_id(9, "jamesle01")

    

    os.makedirs("output_heatmaps", exist_ok=True)
    player_tracks_drawer = PlayerTracksDrawer(player_mapper=player_mapper)
    ball_tracks_drawer = BallTracksDrawer()
    court_keypoint_drawer = CourtKeypointDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    frame_number_drawer = FrameNumberDrawer()
    pass_and_interceptions_drawer = PassInterceptionDrawer()
    tactical_view_drawer = TacticalViewDrawer()
    speed_and_distance_drawer = SpeedAndDistanceDrawer()

    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks, player_assignment, ball_aquisition)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)
    output_video_frames = court_keypoint_drawer.draw(output_video_frames, court_keypoints_per_frame)
    output_video_frames = frame_number_drawer.draw(output_video_frames)
    output_video_frames = team_ball_control_drawer.draw(output_video_frames, player_assignment, ball_aquisition)
    output_video_frames = pass_and_interceptions_drawer.draw(output_video_frames, passes, interceptions)
    output_video_frames = speed_and_distance_drawer.draw(output_video_frames, player_tracks, player_distances_per_frame, player_speed_per_frame)
    output_video_frames = tactical_view_drawer.draw(
        output_video_frames,
        tactical_view_converter.court_image_path,
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.key_points,
        tactical_player_positions,
        player_assignment,
        ball_aquisition
    )

    # Win probability overlay
    output_video_frames = overlay_win_probability_on_frames(output_video_frames, "D:/basketball ml - Copy - Copy/coefs.csv")

    # Heatmap generation
    heat_gen = PlayerHeatmapGenerator(
        court_w=tactical_view_converter.width,
        court_h=tactical_view_converter.height
    )
    for tactical_pos in tactical_player_positions:
        heat_gen.add_frame_positions(tactical_pos)

    all_heatmaps = heat_gen.get_heatmaps()
    for player_id, heat in all_heatmaps.items():
        heat_img = heat_gen.heatmap_to_bgr(heat)
        save_path = os.path.join("output_heatmaps", f"player_{player_id}.jpg")
        cv2.imwrite(save_path, heat_img)

    # Save video
    save_video(output_video_frames, OUTPUT_VIDEO_PATH)

if __name__ == "__main__":
    main()
