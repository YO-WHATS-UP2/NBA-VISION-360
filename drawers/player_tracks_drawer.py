import cv2
from drawers.utils import draw_ellipse, draw_traingle

class PlayerTracksDrawer:
    def __init__(self, team_1_color=[255, 245, 238], team_2_color=[128, 0, 0], player_mapper=None):
        self.default_player_team_id = 1
        self.team_1_color=team_1_color
        self.team_2_color=team_2_color
        self.player_mapper = player_mapper
        

    def draw(self, video_frames, tracks, player_assignment, ball_aquisition):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks[frame_num]
            player_assignment_for_frame = player_assignment[frame_num]
            player_id_has_ball = ball_aquisition[frame_num]
            
            # Draw Players
            for track_id, player in player_dict.items():
                team_id = player_assignment_for_frame.get(track_id, self.default_player_team_id)
                
                if team_id == 1:
                    color = self.team_1_color
                else:
                    color = self.team_2_color
                
                frame = draw_ellipse(frame, player["bbox"], color, track_id)
                
                if track_id == player_id_has_ball:
                    frame = draw_traingle(frame, player["bbox"], (0, 0, 255))
                
                # Draw player name
                if self.player_mapper:
                    player_name = self.player_mapper.get_player_name_from_yolo_id(track_id)
                    if player_name and player_name != "Unknown Player":
                        x1, y1, x2, y2 = player["bbox"]
                        cv2.putText(frame, player_name, (int(x1), int(y1-10)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                       
            # Update animation frame count
            
            output_video_frames.append(frame)
        
        return output_video_frames
        