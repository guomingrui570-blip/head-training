import cv2
import numpy as np
from ultralytics import YOLO
import random

# ================= 遊戲設定參數 =================
CELL_SIZE = 30           # 迷宮格子大小
MAZE_W, MAZE_H = 15, 15  # 迷宮長寬 (必須是奇數)
MAX_HEALTH = 100         # 最大血量

# 休閒模式參數
CASUAL_SPEED_FACTOR = 0.04 
DEADZONE = 30            

# 運動模式參數
SPORTS_EASING = 0.15     # 追蹤平滑係數

# 防穿牆碰撞檢測設定
MAX_STEP_PIXELS = 20.0    # 每次碰撞檢測的最大移動像素 (小於格子寬度，確保不穿牆)

# 狀態常數
STATE_MENU = 0
STATE_CASUAL = 1
STATE_SPORTS = 2

# ================= 迷宮生成演算法 =================
def generate_maze(w, h):
    maze = [[1] * w for _ in range(h)]
    def carve(x, y):
        maze[y][x] = 0
        dirs = [(0, -2), (0, 2), (-2, 0), (2, 0)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 1 <= nx < w - 1 and 1 <= ny < h - 1 and maze[ny][nx] == 1:
                maze[y + dy // 2][x + dx // 2] = 0
                carve(nx, ny)
                
    start_x, start_y = w // 2, h // 2
    carve(start_x, start_y)
    
    maze[h - 2][w - 2] = 2
    maze[h - 2][w - 3] = 0 
    return maze

# ================= 遊戲狀態管理器 =================
class GameState:
    def __init__(self):
        self.state = STATE_MENU
        self.reset()

    def reset(self):
        self.maze = generate_maze(MAZE_W, MAZE_H)
        self.health = MAX_HEALTH
        self.game_over = False
        self.victory = False
        
        # 休閒模式：迷宮基準座標
        center_cell_x, center_cell_y = MAZE_W // 2, MAZE_H // 2
        self.maze_px = 320 - (center_cell_x * CELL_SIZE + CELL_SIZE // 2)
        self.maze_py = 240 - (center_cell_y * CELL_SIZE + CELL_SIZE // 2)

        # 運動模式：迷宮固定在畫面中央
        self.sports_maze_px = (640 - MAZE_W * CELL_SIZE) // 2
        self.sports_maze_py = (480 - MAZE_H * CELL_SIZE) // 2
        self.player_x = 320.0
        self.player_y = 240.0

game = GameState()

# ================= 滑鼠點擊事件 =================
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if game.state == STATE_MENU:
            if 100 <= x <= 300 and 200 <= y <= 280:
                game.state = STATE_CASUAL
                game.reset()
            elif 340 <= x <= 540 and 200 <= y <= 280:
                game.state = STATE_SPORTS
                game.reset()
        else:
            if 520 <= x <= 620 and 20 <= y <= 60:
                game.reset()
            elif 520 <= x <= 620 and 70 <= y <= 110:
                game.state = STATE_MENU

# ================= 主程式 =================
model = YOLO('yolo26n-pose.pt') 

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("YOLO Maze Game")
cv2.setMouseCallback("YOLO Maze Game", mouse_click)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h_frame, w_frame, _ = frame.shape
    center_x, center_y = w_frame // 2, h_frame // 2

    # 1. 主選單
    if game.state == STATE_MENU:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_frame, h_frame), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        cv2.putText(frame, "YOLO MAZE GAME", (120, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        cv2.rectangle(frame, (100, 200), (300, 280), (100, 200, 100), -1)
        cv2.putText(frame, "Casual Mode", (125, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.rectangle(frame, (340, 200), (540, 280), (100, 100, 200), -1)
        cv2.putText(frame, "Sports Mode", (365, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("YOLO Maze Game", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # 2. 辨識鼻子
    results = model(frame, conf=0.5, verbose=False)
    nose_x, nose_y = center_x, center_y 
    
    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        keypoints = results[0].keypoints.xy[0]
        if len(keypoints) > 0:
            nx, ny = int(keypoints[0][0]), int(keypoints[0][1])
            if nx != 0 and ny != 0: 
                nose_x, nose_y = nx, ny
                glow_nose = frame.copy()
                cv2.circle(glow_nose, (nose_x, nose_y), 16, (100, 255, 100), -1)
                frame = cv2.addWeighted(glow_nose, 0.7, frame, 0.3, 0)
                cv2.circle(frame, (nose_x, nose_y), 8, (100, 255, 100), -1)
                cv2.circle(frame, (nose_x, nose_y), 8, (255, 255, 255), 1)

    # 遊戲邏輯與防穿牆碰撞處理
    if not game.game_over and not game.victory:
        if game.state == STATE_CASUAL:
            dx = nose_x - center_x
            dy = nose_y - center_y
            if abs(dx) < DEADZONE: dx = 0
            if abs(dy) < DEADZONE: dy = 0

            # 計算總目標移動量
            move_x = -dx * CASUAL_SPEED_FACTOR
            move_y = -dy * CASUAL_SPEED_FACTOR
            dist = np.hypot(move_x, move_y)

            if dist > 0:
                steps = int(np.ceil(dist / MAX_STEP_PIXELS))
                step_x = move_x / steps
                step_y = move_y / steps
                
                curr_px, curr_py = game.maze_px, game.maze_py
                hit_wall = False

                for _ in range(steps):
                    next_px = curr_px + step_x
                    next_py = curr_py + step_y
                    
                    rel_x = center_x - next_px
                    rel_y = center_y - next_py
                    grid_x, grid_y = int(rel_x // CELL_SIZE), int(rel_y // CELL_SIZE)

                    if 0 <= grid_x < MAZE_W and 0 <= grid_y < MAZE_H:
                        cell_val = game.maze[grid_y][grid_x]
                        if cell_val == 1:
                            hit_wall = True
                            break
                        elif cell_val == 2:
                            game.victory = True
                            curr_px, curr_py = next_px, next_py
                            break
                        else:
                            curr_px, curr_py = next_px, next_py
                    else:
                        hit_wall = True
                        break

                if hit_wall:
                    game.health -= 2
                    if game.health <= 0: game.game_over = True
                
                game.maze_px, game.maze_py = curr_px, curr_py

        elif game.state == STATE_SPORTS:
            # 運動模式：計算紅點要移向鼻子的總位移
            target_x = game.player_x + (nose_x - game.player_x) * SPORTS_EASING
            target_y = game.player_y + (nose_y - game.player_y) * SPORTS_EASING
            
            move_x = target_x - game.player_x
            move_y = target_y - game.player_y
            dist = np.hypot(move_x, move_y)

            if dist > 0:
                steps = int(np.ceil(dist / MAX_STEP_PIXELS))
                step_x = move_x / steps
                step_y = move_y / steps
                
                curr_x, curr_y = game.player_x, game.player_y
                hit_wall = False

                for _ in range(steps):
                    next_x = curr_x + step_x
                    next_y = curr_y + step_y
                    
                    rel_x = next_x - game.sports_maze_px
                    rel_y = next_y - game.sports_maze_py
                    grid_x, grid_y = int(rel_x // CELL_SIZE), int(rel_y // CELL_SIZE)

                    if 0 <= grid_x < MAZE_W and 0 <= grid_y < MAZE_H:
                        cell_val = game.maze[grid_y][grid_x]
                        if cell_val == 1:
                            hit_wall = True
                            break
                        elif cell_val == 2:
                            game.victory = True
                            curr_x, curr_y = next_x, next_y
                            break
                        else:
                            curr_x, curr_y = next_x, next_y
                    else:
                        hit_wall = True
                        break

                if hit_wall:
                    game.health -= 2
                    if game.health <= 0: game.game_over = True
                
                game.player_x, game.player_y = curr_x, curr_y

    # 3. 繪製遊戲畫面
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w_frame, h_frame), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cur_maze_px = game.maze_px if game.state == STATE_CASUAL else game.sports_maze_px
    cur_maze_py = game.maze_py if game.state == STATE_CASUAL else game.sports_maze_py

    # 畫迷宮
    for y in range(MAZE_H):
        for x in range(MAZE_W):
            cell_type = game.maze[y][x]
            if cell_type != 0:
                draw_x = int(cur_maze_px + x * CELL_SIZE)
                draw_y = int(cur_maze_py + y * CELL_SIZE)
                if -CELL_SIZE < draw_x < w_frame and -CELL_SIZE < draw_y < h_frame:
                    if cell_type == 1:
                        cv2.rectangle(frame, (draw_x, draw_y), (draw_x + CELL_SIZE, draw_y + CELL_SIZE), (200, 200, 200), 2)
                    elif cell_type == 2:
                        cv2.rectangle(frame, (draw_x, draw_y), (draw_x + CELL_SIZE, draw_y + CELL_SIZE), (0, 255, 255), -1)

    # 畫玩家紅點
    p_x = center_x if game.state == STATE_CASUAL else int(game.player_x)
    p_y = center_y if game.state == STATE_CASUAL else int(game.player_y)
    cv2.circle(frame, (p_x, p_y), 12, (0, 0, 255), -1)
    cv2.circle(frame, (p_x, p_y), 12, (255, 255, 255), 2)

    # 繪製 UI (血條與按鈕)
    cv2.rectangle(frame, (20, 20), (220, 40), (0, 0, 150), -1)
    health_width = int((game.health / MAX_HEALTH) * 200)
    if health_width > 0:
        cv2.rectangle(frame, (20, 20), (20 + health_width, 40), (0, 255, 0), -1)
    cv2.putText(frame, f"HP: {game.health}", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Reset 按鈕
    cv2.rectangle(frame, (520, 20), (620, 60), (255, 100, 100), -1)
    cv2.putText(frame, "Reset", (540, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Menu 按鈕
    cv2.rectangle(frame, (520, 70), (620, 110), (100, 100, 255), -1)
    cv2.putText(frame, "Menu", (545, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    if game.game_over:
        cv2.putText(frame, "GAME OVER", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    elif game.victory:
        cv2.putText(frame, "VICTORY!", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)

    cv2.imshow("YOLO Maze Game", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()