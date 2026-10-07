import cv2
import numpy as np
from ultralytics import YOLO
import random

# ================= 畫面 / 顯示適應相關 =================
# 【重要】未來要移植到網頁：所有遊戲邏輯座標統一在 VIRTUAL_W/H = 640x480 的虛擬座標系計算
#        最終顯示時再透過 scale_virtual_xy() 等函數映射到使用者真實畫面尺寸
#        如此邏輯完全不依賴畫面解析度，網頁端 css transform scale() / canvas 大小直接換算即可
VIRTUAL_W, VIRTUAL_H = 960, 540  # 改為 16:9（1080p 的 1/2，網頁/手機屏幕主流比例）
# 攝影機透明度（極簡像素風建議 0.30 ~ 0.45，能看到自己 + 不會影響迷宮辨識）
CAMERA_ALPHA = 0.35
# 全螢幕開關熱鍵：按 F 或 F11
TOGGLE_FULLSCREEN_KEYS = [ord('f'), ord('F')]

# ================= 極簡像素風（Minimalist Pixel Art）主題 =================
# 黑白灰為主 + 4px 像素硬邊 + 無圓角無漸變無陰影
#   視覺語言：
#     背景        = 純白 (#FCFCFC) 方便後面人物透明顯示
#     線條/文字    = 近黑  (#1A1A1A)，線條粗細統一 4px (虛擬像素)
#     弱調邊框    = 中灰  (#8A8A8A)，例如按鈕未選中、HP 背景
#     點綴色      = 極少，僅玩家/出口少量使用 #222 對比
#     UI 卡片     = 白底 90% 不透明（能看到後面人臉）+ 4px 純黑方框
PIXEL_BORDER_VIRT = 4  # 虛擬系：所有線條 / 按鈕邊框 4px
THEME = {
    # 背景（極簡純白，方便讓後面攝像頭人物透出來）
    'bg': (252, 252, 252),
    # 迷宮牆體：4px 黑像素線條填充 + 中灰底色（不描邊，填色更像素）
    'maze_wall_fill':   (218, 218, 218),
    'maze_wall_border': ( 26,  26,  26),   # 4px 像素黑框
    'maze_path_fill':   (250, 250, 250),   # 通道近白
    # 出口（極簡：黑 4px 框 + 棋盤格）
    'exit_fill':        (240, 240, 240),
    'exit_border':      ( 26,  26,  26),
    # 玩家球（極簡像素：黑 4px 框 + 深灰填充 + 左上 2px 白點表示"高光"）
    'player_inner':     ( 52,  52,  52),
    'player_outer':     (250, 250, 250),
    'player_border':    ( 26,  26,  26),
    # 鼻子偵測標記（極簡 4px 黑框 + 半透明白圓）
    'nose_inner':       (250, 250, 250),
    'nose_outer':       ( 26,  26,  26),
    # UI 卡片（白底 90% 不透明 + 4px 黑框，無圓角無陰影）
    'card_fill':        (255, 255, 255),   # 後續用 addWeighted 做 0.9 透明
    'card_shadow':      None,              # 極簡：無陰影
    'card_border':      ( 26,  26,  26),
    'card_alpha':       0.90,              # 卡片 90% 不透明，能看到後面人臉
    # 血條（HP：黑 4px 框 + 按血量分 3 段灰階）
    'hp_bg':            (232, 232, 232),
    'hp_good':          ( 86,  86,  86),   # >60% 深灰
    'hp_mid':           (138, 138, 138),   # >30% 中灰
    'hp_low':           (200,  80,  80),   # <30% 微紅點綴（保持極簡只用 1 個暖色）
    # Reset / Menu 按鈕（4px 黑框 + 淺灰填充，hover/點擊=深灰）
    'btn_reset_fill':   (242, 242, 242),
    'btn_reset_txt':    ( 26,  26,  26),
    'btn_menu_fill':    (242, 242, 242),
    'btn_menu_txt':     ( 26,  26,  26),
    # 主選單按鈕
    'menu_casual_fill': (246, 246, 246),
    'menu_casual_txt':  ( 26,  26,  26),
    'menu_sports_fill': (246, 246, 246),
    'menu_sports_txt':  ( 26,  26,  26),
    # 文本（全黑，字重粗，像素點感）
    'txt_title':        ( 26,  26,  26),
    'txt_sub':          (108, 108, 108),
    # 勝利/失敗（極簡大卡片 4px 黑框，文字粗黑）
    'gameover_fill':    (248, 248, 248),
    'gameover_txt':     ( 26,  26,  26),
    'victory_fill':     (248, 248, 248),
    'victory_txt':      ( 26,  26,  26),
}

# ================= 遊戲設定參數 =================
CELL_SIZE = 30           # 迷宮格子大小（虛擬座標系）
MAZE_W, MAZE_H = 15, 15  # 迷宮長寬 (必須是奇數)
MAX_HEALTH = 250         # 最大血量

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

# ================= 全螢幕 / 縮放適應 =================
display = {
    'W': VIRTUAL_W,           # 目前實際顯示畫面 width
    'H': VIRTUAL_H,           # 目前實際顯示畫面 height
    'scale': 1.0,             # virtual → display 的等比縮放係數（取 min(dW/VW, dH/VH)）
    'offX': 0,                # letterbox 時的左邊 padding 像素
    'offY': 0,                # letterbox 時的上邊 padding 像素
    'fullscreen': False,      # 是否全螢幕模式
    'window_name': "YOLO Maze Game",
}
def update_display_geometry(window_w: int, window_h: int):
    """根據目前視窗/螢幕大小，更新 display 的 scale/offX/offY"""
    display['W'] = max(1, window_w)
    display['H'] = max(1, window_h)
    scale = min(display['W'] / VIRTUAL_W, display['H'] / VIRTUAL_H)
    display['scale'] = scale
    # letterbox 置中
    display['offX'] = int((display['W'] - VIRTUAL_W * scale) / 2)
    display['offY'] = int((display['H'] - VIRTUAL_H * scale) / 2)

def scale_virtual_xy(vx: float, vy: float):
    """虛擬座標 → 顯示座標"""
    return int(display['offX'] + vx * display['scale']), \
           int(display['offY'] + vy * display['scale'])

def scale_val(v: float):
    """虛擬長度 → 顯示長度"""
    return max(1, int(v * display['scale']))

# 將滑鼠點擊（顯示座標）反算回虛擬座標，供選單/按鈕命中檢查使用
def unscale_display_xy(dx: int, dy: int):
    vx = (dx - display['offX']) / display['scale'] if display['scale'] > 0 else 0
    vy = (dy - display['offY']) / display['scale'] if display['scale'] > 0 else 0
    return vx, vy

# OpenCV 視窗大小改變 callback（Windows 版 highgui 可選，不支援時也沒關係）
def _window_resize_cb(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEHWHEEL or event == cv2.EVENT_MOUSEWHEEL:
        return
    try:
        _retval = cv2.getWindowImageRect(display['window_name'])
        if _retval and len(_retval) == 4:
            _, _, ww, wh = _retval
            if ww > 10 and wh > 10:
                update_display_geometry(ww, wh)
    except Exception:
        pass

def toggle_fullscreen():
    display['fullscreen'] = not display['fullscreen']
    if display['fullscreen']:
        # 先切全螢幕讓視窗取得畫面大小
        cv2.setWindowProperty(display['window_name'],
                              cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        # Windows 下用 getWindowImageRect 拿實際尺寸；macOS/linux fallback 用 primary = 1920x1080
        try:
            r = cv2.getWindowImageRect(display['window_name'])
            if r and r[2] > 10 and r[3] > 10:
                update_display_geometry(r[2], r[3])
            else:
                update_display_geometry(1920, 1080)
        except Exception:
            update_display_geometry(1920, 1080)
    else:
        cv2.setWindowProperty(display['window_name'],
                              cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_NORMAL)
        update_display_geometry(VIRTUAL_W, VIRTUAL_H)
        cv2.resizeWindow(display['window_name'], VIRTUAL_W, VIRTUAL_H)

# ================= 繪圖工具（極簡像素風） =================
# 美術準則：
#   1. 全部矩形硬邊 (r=0)、4px 黑框、不抗鋸齒 (lineType=cv2.LINE_4)
#   2. 文字厚度統一用 PIXEL_BORDER_VIRT (4 虛擬像素 = 縮放後也清晰
#   3. UI 卡片先做 90% 不透明 (card_alpha)，永遠能看到後面鏡頭人像

def draw_rect_fill_border(img, fx1, fy1, fx2, fy2, fill, border, border_virt_px=PIXEL_BORDER_VIRT):
    """純填充 + 像素黑框（直接 4 虛擬像素邊框）"""
    cv2.rectangle(img, (fx1, fy1), (fx2, fy2), fill, -1, lineType=cv2.LINE_4)
    bw = scale_val(border_virt_px)
    if bw > 0:
        cv2.rectangle(img, (fx1, fy1), (fx2, fy2), border, bw, lineType=cv2.LINE_4)

def draw_card(img, vx1, vy1, vx2, vy2, color_fill=None, color_border=None,
              shadow=False, r=0, alpha=None):
    """極簡風卡片：白底 + 4px 黑框 + alpha 0.90 半透明，確保後面人像可見
       shadow/r 參數保留只是為了兼容舊調用，實際永遠 r=0, shadow=False"""
    if color_fill   is None: color_fill   = THEME['card_fill']
    if color_border is None: color_border = THEME['card_border']
    if alpha is None: alpha = THEME.get('card_alpha', 1.0)
    fx1, fy1 = scale_virtual_xy(vx1, vy1)
    fx2, fy2 = scale_virtual_xy(vx2, vy2)
    # 在子畫面做半透明填充（維持人臉顯示）
    if 0.0 < alpha < 1.0 and (fx2-fx1 > 4) and (fy2-fy1 > 4):
        sub = img[fy1:fy2, fx1:fx2]
        ovr = np.full_like(sub, color_fill, dtype=np.uint8)
        img[fy1:fy2, fx1:fx2] = cv2.addWeighted(sub, 1.0-alpha, ovr, alpha, 0.0)
    else:
        cv2.rectangle(img, (fx1, fy1), (fx2, fy2), color_fill, -1, lineType=cv2.LINE_4)
    # 畫黑框
    bw = scale_val(PIXEL_BORDER_VIRT)
    if bw > 0:
        cv2.rectangle(img, (fx1, fy1), (fx2, fy2), color_border, bw, lineType=cv2.LINE_4)
    return (fx1, fy1, fx2, fy2)  # 回傳顯示座標供文字定位

def put_text(img, text, vx, vy, size=0.7, color=None, thickness=2, anchor='tl'):
    if color is None: color = THEME['txt_title']
    fx, fy = scale_virtual_xy(vx, vy)
    s = size * display['scale']
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, thickness)
    if   anchor == 'tr': fx -= tw
    elif anchor == 'bl': fy += th
    elif anchor == 'tc': fx -= tw // 2
    elif anchor == 'bc': fx -= tw // 2; fy += th
    # 像素字：字體黑體加粗 1，抗鋸齒 LINE_4 (像素感
    cv2.putText(img, text, (fx, fy + baseline),
                cv2.FONT_HERSHEY_SIMPLEX, s, color, thickness, lineType=cv2.LINE_4)

def draw_circle_pixel(img, vcx, vcy, vr, fill_color=None, border_color=None):
    cx, cy = scale_virtual_xy(vcx, vcy)
    r = scale_val(vr)
    if fill_color is not None:
        cv2.circle(img, (cx, cy), r, fill_color, -1, lineType=cv2.LINE_4)
    if border_color is not None:
        bw = max(1, scale_val(PIXEL_BORDER_VIRT))
        cv2.circle(img, (cx, cy), r, border_color, bw, lineType=cv2.LINE_4)
    return cx, cy, r

def draw_maze(img, cur_maze_px, cur_maze_py):
    """迷宮：格子直接黑白灰像素畫，保證後面人臉不會擋住
    【注意】為了"迷宮不擋人物 → 先做 0.72 alpha 填色（只有牆 72% 黑框 + 出口棋盤
"""
    for y in range(MAZE_H):
        for x in range(MAZE_W):
            cell_type = game.maze[y][x]
            vx = cur_maze_px + x * CELL_SIZE
            vy = cur_maze_py + y * CELL_SIZE
            fx1, fy1 = scale_virtual_xy(vx, vy)
            fx2, fy2 = scale_virtual_xy(vx + CELL_SIZE, vy + CELL_SIZE)
            if fx2 < 0 or fy2 < 0 or fx1 >= display['W'] or fy1 >= display['H']:
                continue
            if cell_type == 0:
                # 通道：極簡：幾乎透明的淺灰（不擋人物）
                sub = img[fy1:fy2, fx1:fx2]
                if sub.size:
                    ovr = np.full_like(sub, THEME['maze_path_fill'], dtype=np.uint8)
                    img[fy1:fy2, fx1:fx2] = cv2.addWeighted(sub, 0.92, ovr, 0.08, 0.0)
            elif cell_type == 1:
                # 牆：填色 0.80 alpha + 4px 黑框（確保導航能讀迷宮
                sub = img[fy1:fy2, fx1:fx2]
                if sub.size:
                    ovr = np.full_like(sub, THEME['maze_wall_fill'], dtype=np.uint8)
                    img[fy1:fy2, fx1:fx2] = cv2.addWeighted(sub, 0.30, ovr, 0.70, 0.0)
                bw = scale_val(PIXEL_BORDER_VIRT)
                if bw > 0:
                    cv2.rectangle(img, (fx1, fy1), (fx2, fy2), THEME['maze_wall_border'], bw, lineType=cv2.LINE_4)
            elif cell_type == 2:
                # 出口：棋盤黑點綴（填半透明（迷宮
                sub = img[fy1:fy2, fx1:fx2]
                if sub.size:
                    ovr = np.full_like(sub, THEME['exit_fill'], dtype=np.uint8)
                    img[fy1:fy2, fx1:fx2] = cv2.addWeighted(sub, 0.35, ovr, 0.65, 0.0)
                # 4px 黑框
                bw = scale_val(PIXEL_BORDER_VIRT)
                if bw > 0:
                    cv2.rectangle(img, (fx1, fy1), (fx2, fy2), THEME['exit_border'], bw, lineType=cv2.LINE_4)
                # 中央"目標圓點（極簡提示出口
                cfx, cfy = (fx1+fx2)//2, (fy1+fy2)//2
                rr = max(2, int(min(fx2-fx1, fy2-fy1) * 0.18))
                cv2.circle(img, (cfx,cfy), rr, THEME['exit_border'], -1, lineType=cv2.LINE_4)
    # 迷宮整體外框（額外 4px 黑框）
    fx1, fy1 = scale_virtual_xy(cur_maze_px - PIXEL_BORDER_VIRT,
                                cur_maze_py - PIXEL_BORDER_VIRT)
    fx2, fy2 = scale_virtual_xy(cur_maze_px + MAZE_W * CELL_SIZE + PIXEL_BORDER_VIRT,
                                cur_maze_py + MAZE_H * CELL_SIZE + PIXEL_BORDER_VIRT)
    bw = scale_val(PIXEL_BORDER_VIRT)
    if bw > 0:
        cv2.rectangle(img, (fx1, fy1), (fx2, fy2), THEME['card_border'], bw, lineType=cv2.LINE_4)

def draw_player(img, vx, vy, hit_flash=False):
    """玩家：極簡像素風雙色半透明 + 4px 黑框"""
    (cx, cy) = scale_virtual_xy(vx, vy)
    r = scale_val(14)
    if r <= 0: return
    x1, y1, x2, y2 = cx-r, cy-r, cx+r, cy+r
    if x2-x1 > 0 and y2-y1 > 0:
        sub = img[y1:y2, x1:x2]
        if sub.size:
            c_fill = THEME['gameover_txt'] if hit_flash else THEME['player_outer']
            ovr = np.full_like(sub, c_fill, dtype=np.uint8)
            img[y1:y2, x1:x2] = cv2.addWeighted(sub, 0.25, ovr, 0.75, 0.0)
    # 4px 黑框
    bw = max(1, scale_val(PIXEL_BORDER_VIRT))
    cv2.circle(img, (cx, cy), r, THEME['player_border'], bw, lineType=cv2.LINE_4)
    # 內部深灰填充
    r_in = max(2, r - bw - 1)
    c_in = THEME['gameover_fill'] if hit_flash else THEME['player_inner']
    cv2.circle(img, (cx, cy), r_in, c_in, -1, lineType=cv2.LINE_4)
    # 左上 2 px 高光點
    hl = max(1, scale_val(3))
    hx, hy = cx - int(r*0.40), cy - int(r*0.40)
    cv2.circle(img, (hx, hy), hl, (240,240,240), -1, lineType=cv2.LINE_4)

def draw_nose_marker(img, vx, vy):
    """鼻子標記：極簡 4px 黑框 + 半透明白圈 0.60 alpha 不擋人臉"""
    cx, cy = scale_virtual_xy(vx, vy)
    r_out = scale_val(15)
    r_in  = scale_val(7)
    # 外圈：半透明白
    if r_out > 0:
        overlay = np.zeros_like(img, dtype=np.uint8)
        cv2.circle(overlay, (cx, cy), r_out, THEME['nose_inner'], -1, lineType=cv2.LINE_4)
        img[:] = cv2.addWeighted(img, 1.0, overlay, 0.55, 0)
    # 實心黑 / 16 進階 4px 黑框
    if r_in > 0:
        cv2.circle(img, (cx, cy), r_in, THEME['nose_outer'], -1, lineType=cv2.LINE_4)
    bw = max(1, scale_val(PIXEL_BORDER_VIRT))
    cv2.circle(img, (cx, cy), r_in, (250,250,250), bw, lineType=cv2.LINE_4)

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
        
        # 休閒模式：迷宮基準座標（16:9 960x540 中心 = 480,270）
        center_cell_x, center_cell_y = MAZE_W // 2, MAZE_H // 2
        self.maze_px = 480 - (center_cell_x * CELL_SIZE + CELL_SIZE // 2)
        self.maze_py = 270 - (center_cell_y * CELL_SIZE + CELL_SIZE // 2)

        # 運動模式：迷宮固定在 16:9 畫面中央
        self.sports_maze_px = (960 - MAZE_W * CELL_SIZE) // 2
        self.sports_maze_py = (540 - MAZE_H * CELL_SIZE) // 2
        self.player_x = 480.0
        self.player_y = 270.0

game = GameState()

# ================= 滑鼠點擊事件 =================
# 注意：x/y 為「顯示座標」（全螢幕/視窗大小）→ 先 unscale 回 VIRTUAL_W/H 虛擬座標再做命中
def mouse_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    # 先讓 resize callback 更新一次 scale（防止縮放後點擊座標偏移）
    try:
        r = cv2.getWindowImageRect(display['window_name'])
        if r and r[2] > 10 and r[3] > 10:
            update_display_geometry(r[2], r[3])
    except Exception:
        pass
    vx, vy = unscale_display_xy(x, y)
    if game.state == STATE_MENU:
        # Casual / Sports 按鈕（16:9 960 畫面中央，256x72 每個）
        if  272 <= vx <=  472 and 230 <= vy <= 310:
            game.state = STATE_CASUAL
            game.reset()
        elif 488 <= vx <= 688 and 230 <= vy <= 310:
            game.state = STATE_SPORTS
            game.reset()
    else:
        # Reset / Menu 按鈕（16:9 右上角 800..928）
        if 800 <= vx <= 928 and  22 <= vy <=  62:
            game.reset()
        elif 800 <= vx <= 928 and 72 <= vy <= 112:
            game.state = STATE_MENU

# ================= 主程式 =================
# 注：你本地應該是 yolov8n-pose.pt (與 nose_track_fps 對齊)，如確實有 yolo26n-pose.pt 再改回來
MODEL_PATH = 'yolov8n-pose.pt'
try:
    model = YOLO(MODEL_PATH)
except Exception:
    # 兼容舊模型名
    try:
        model = YOLO('yolo26n-pose.pt')
    except Exception as e:
        raise RuntimeError(f'找不到姿態模型，請確認 {MODEL_PATH}（或 yolo26n-pose.pt）位於腳本同目錄：{e}')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 啟用可縮放視窗（讓用戶拉邊、按 F 切全螢幕）
cv2.namedWindow(display['window_name'], cv2.WINDOW_NORMAL)
cv2.resizeWindow(display['window_name'], VIRTUAL_W, VIRTUAL_H)
cv2.setMouseCallback(display['window_name'], mouse_click)
# 開啟高 DPI 縮放（部分 OpenCV build 支援）
try:
    cv2.setWindowProperty(display['window_name'], cv2.WND_PROP_TOPMOST, 0)
except Exception:
    pass

# 撞牆閃爍計時器（毫秒，display 毫秒級，用 performance counter）
import time as _time
_last_hit_t = 0.0

def start_hit_flash():
    global _last_hit_t
    _last_hit_t = _time.perf_counter()

def is_hit_flash_active():
    # 閃紅 220ms
    return (_time.perf_counter() - _last_hit_t) < 0.22

while True:
    success, frame = cap.read()
    if not success:
        break

    # 相機影像固定 resize 到虛擬尺寸 640×480，作為後續 "30% 不透明背景" 素材
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (VIRTUAL_W, VIRTUAL_H), interpolation=cv2.INTER_AREA)
    h_frame, w_frame, _ = frame.shape  # = VIRTUAL_H, VIRTUAL_W
    # 虛擬系中心（遊戲邏輯用，固定不變，不受縮放影響）
    center_x, center_y = w_frame // 2, h_frame // 2

    # ================= 構建最終顯示畫面（display['W'] x display['H'] 真實大小） =================
    # 1) 淺米白背景填滿（未來網頁端：body { background:#f4f8fa }）
    try:
        r = cv2.getWindowImageRect(display['window_name'])
        if r and r[2] > 10 and r[3] > 10:
            update_display_geometry(r[2], r[3])
    except Exception:
        pass
    # 真實畫面 canvas
    canvas = np.full((display['H'], display['W'], 3), THEME['bg'], dtype=np.uint8)
    # 2) 攝像頭【全螢幕鋪滿整個 canvas】做 35% alpha 混合（人物永遠在最底層，迷宮/UI 疊在上面不擋住人物）
    #    【關鍵】之前只縮到迷宮 letterbox 區（導致人物只在中間），現在直接鋪到 display 全尺寸
    cam_w_scaled = display['W']
    cam_h_scaled = display['H']
    if cam_w_scaled > 0 and cam_h_scaled > 0:
        # INTER_AREA 縮放比較銳利，配合像素風剛好
        cam_resized = cv2.resize(frame, (cam_w_scaled, cam_h_scaled),
                                 interpolation=cv2.INTER_AREA)
        # 全畫面 blend：米白背景 1-35% + 攝像頭人像 35%（調 CAMERA_ALPHA 控制不透明度
        canvas[:] = cv2.addWeighted(canvas,     1.0 - CAMERA_ALPHA,
                                    cam_resized,       CAMERA_ALPHA, 0.0)

    # 1. 主選單（16:9 960 畫面，所有座標重新調整到 960x540 虛擬系）
    if game.state == STATE_MENU:
        # 標題卡片（16:9 畫面中央偏上）
        draw_card(canvas, 260, 60, 700, 150,
                  color_fill=THEME['card_fill'])
        put_text(canvas, "YOLO MAZE",
                 480, 105, size=1.8, color=THEME['txt_title'], thickness=5, anchor='tc')
        put_text(canvas, "F = Fullscreen | 1 Casual | 2 Sports | Q Quit",
                 480, 135, size=0.58, color=THEME['txt_sub'], thickness=2, anchor='tc')
        # Casual 卡片（中央左）
        draw_card(canvas, 272, 230, 472, 310,
                  color_fill=THEME['menu_casual_fill'],
                  color_border=THEME['menu_casual_txt'])
        put_text(canvas, "Casual", 372, 275, size=0.95,
                 color=THEME['menu_casual_txt'], thickness=4, anchor='tc')
        # Sports 卡片（中央右）
        draw_card(canvas, 488, 230, 688, 310,
                  color_fill=THEME['menu_sports_fill'],
                  color_border=THEME['menu_sports_txt'])
        put_text(canvas, "Sports", 588, 275, size=0.95,
                 color=THEME['menu_sports_txt'], thickness=4, anchor='tc')
        # 底部提示（一行短寫）
        put_text(canvas, "Control the ball with your head.",
                 480, 470, size=0.65, color=THEME['txt_sub'], thickness=2, anchor='tc')

        cv2.imshow(display['window_name'], canvas)
        _k = cv2.waitKey(1) & 0xFF
        if _k == ord('q'):
            break
        if _k in TOGGLE_FULLSCREEN_KEYS:
            toggle_fullscreen()
        if _k == ord('1'):
            game.state = STATE_CASUAL ; game.reset()
        if _k == ord('2'):
            game.state = STATE_SPORTS ; game.reset()
        continue

    # 2. 辨識鼻子（注意：推理在 640x480 虛擬尺寸 frame 上做，座標直接對應虛擬系）
    results = model(frame, conf=0.5, verbose=False)
    nose_x, nose_y = center_x, center_y

    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        keypoints = results[0].keypoints.xy[0]
        if len(keypoints) > 0:
            nx, ny = float(keypoints[0][0]), float(keypoints[0][1])
            if abs(nx) > 1e-4 and abs(ny) > 1e-4:
                nose_x, nose_y = int(nx), int(ny)
                # 鼻子發光圓（在 canvas 上，注意這裡座標也是虛擬系）
                draw_nose_marker(canvas, nose_x, nose_y)

    # 遊戲邏輯與防穿牆碰撞處理（邏輯座標全部保持在 VIRTUAL_W/H，不隨畫面縮放改變）
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
                    start_hit_flash()
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
                    start_hit_flash()
                    if game.health <= 0: game.game_over = True

                game.player_x, game.player_y = curr_x, curr_y

    # 3. 繪製遊戲畫面（所有元素虛擬座標 → scale 到真實畫面）
    cur_maze_px = game.maze_px if game.state == STATE_CASUAL else game.sports_maze_px
    cur_maze_py = game.maze_py if game.state == STATE_CASUAL else game.sports_maze_py
    # 3.1 迷宮
    draw_maze(canvas, cur_maze_px, cur_maze_py)
    # 3.2 玩家紅點
    if game.state == STATE_CASUAL:
        draw_player(canvas, center_x, center_y, hit_flash=is_hit_flash_active())
    else:
        draw_player(canvas, game.player_x, game.player_y, hit_flash=is_hit_flash_active())

    # 3.3 繪製 UI (左上角 HP 卡片 + 模式標籤，16:9 960 佈局，兩張卡片左右強制對齊)
    # 統一卡片左右邊界：x 從 28 到 300（寬 272 虛擬像素，HP 卡片 + Casual 標籤共用此寬度，左右完全齊）
    HP_CARD_X1, HP_CARD_X2 = 28, 300
    TAG_CARD_X1, TAG_CARD_X2 = HP_CARD_X1, HP_CARD_X2

    # HP 卡片（內部兩行：第一行 HP 文字；第二行 HP 進度條，全部在卡片內 padding 不溢出）
    draw_card(canvas, HP_CARD_X1, 22, HP_CARD_X2, 78)
    HP_PAD_L = 14  # 內部左 padding (虛擬像素)
    HP_PAD_T = 10  # 上 padding
    HP_PAD_B = 10  # 下 padding
    HP_TEXT_Y = 22 + HP_PAD_T
    HP_BAR_Y_TOP = 22 + 28  # 文字高約 14 ~ 16，所以 22+10+16 = 22+26 取 28
    HP_BAR_Y_BOT = 78 - HP_PAD_B
    HP_BAR_X_L = HP_CARD_X1 + HP_PAD_L
    HP_BAR_X_R = HP_CARD_X2 - HP_PAD_L
    # HP 文字（在卡片內部頂部，不超過卡片上邊界）
    put_text(canvas, f"HP {game.health} / {MAX_HEALTH}",
             HP_BAR_X_L, HP_TEXT_Y,
             size=0.65, color=THEME['txt_title'], thickness=2, anchor='tl')
    # HP 背景條（在卡片內部底部 padding 內）
    hp_bg_fx1, hp_bg_fy1 = scale_virtual_xy(HP_BAR_X_L, HP_BAR_Y_TOP)
    hp_bg_fx2, hp_bg_fy2 = scale_virtual_xy(HP_BAR_X_R, HP_BAR_Y_BOT)
    cv2.rectangle(canvas, (hp_bg_fx1, hp_bg_fy1), (hp_bg_fx2, hp_bg_fy2),
                  THEME['hp_bg'], -1, lineType=cv2.LINE_4)
    hp_bw = max(1, scale_val(2))
    cv2.rectangle(canvas, (hp_bg_fx1, hp_bg_fy1), (hp_bg_fx2, hp_bg_fy2),
                  THEME['card_border'], hp_bw, lineType=cv2.LINE_4)
    # HP 前景（根據 HP 換色，極簡灰階）
    bar_total_w_virt = HP_BAR_X_R - HP_BAR_X_L
    health_w_virt = int((game.health / MAX_HEALTH) * bar_total_w_virt)
    if   game.health > MAX_HEALTH * 0.6: hp_color = THEME['hp_good']
    elif game.health > MAX_HEALTH * 0.3: hp_color = THEME['hp_mid']
    else:                                hp_color = THEME['hp_low']
    if health_w_virt > 0:
        hp_virt_end_x = HP_BAR_X_L + health_w_virt
        hp_fx1, hp_fy1 = scale_virtual_xy(HP_BAR_X_L, HP_BAR_Y_TOP)
        hp_fx2, hp_fy2 = scale_virtual_xy(hp_virt_end_x, HP_BAR_Y_BOT)
        if hp_fx2 > hp_fx1:
            cv2.rectangle(canvas, (hp_fx1, hp_fy1), (hp_fx2, hp_fy2),
                          hp_color, -1, lineType=cv2.LINE_4)

    # 3.4 模式標籤（左上角 HP 卡片下方，左右邊界與 HP 卡片完全對齊 28..300）
    # 與 HP 卡片底部 y=78 保持 12 虛擬像素間距：y 從 90 到 122
    tag_color = THEME['menu_casual_txt'] if game.state == STATE_CASUAL else THEME['menu_sports_txt']
    draw_card(canvas, TAG_CARD_X1, 90, TAG_CARD_X2, 122)
    put_text(canvas, ("Casual" if game.state == STATE_CASUAL else "Sports"),
             TAG_CARD_X1 + HP_PAD_L, 104,
             size=0.72, color=tag_color, thickness=3, anchor='tl')

    # 3.5 Reset / Menu 按鈕（16:9 右上角 800..928）
    draw_card(canvas, 800, 22, 928, 62,
              color_fill=THEME['btn_reset_fill'], color_border=THEME['btn_reset_txt'])
    put_text(canvas, "Reset", 864, 44, size=0.65,
             color=THEME['btn_reset_txt'], thickness=2, anchor='tc')
    draw_card(canvas, 800, 72, 928, 112,
              color_fill=THEME['btn_menu_fill'], color_border=THEME['btn_menu_txt'])
    put_text(canvas, "Menu", 864, 94, size=0.65,
             color=THEME['btn_menu_txt'], thickness=2, anchor='tc')

    # 3.6 勝利 / 失敗 大卡片（16:9 畫面中央）
    if game.game_over:
        draw_card(canvas, 280, 130, 680, 410,
                  color_fill=THEME['gameover_fill'],
                  color_border=THEME['gameover_txt'])
        put_text(canvas, "GAME OVER", 480, 220,
                 size=2.3, color=THEME['gameover_txt'], thickness=6, anchor='tc')
        put_text(canvas, "R to retry", 480, 330, size=0.8, color=THEME['txt_sub'], thickness=2, anchor='tc')
    elif game.victory:
        draw_card(canvas, 280, 130, 680, 410,
                  color_fill=THEME['victory_fill'],
                  color_border=THEME['victory_txt'])
        put_text(canvas, "VICTORY", 480, 220,
                 size=2.3, color=THEME['victory_txt'], thickness=6, anchor='tc')
        put_text(canvas, "R for new maze", 480, 330, size=0.8, color=THEME['txt_sub'], thickness=2, anchor='tc')

    cv2.imshow(display['window_name'], canvas)

    _k = cv2.waitKey(1) & 0xFF
    if _k == ord('q'):
        break
    if _k in TOGGLE_FULLSCREEN_KEYS:
        toggle_fullscreen()
    if _k == ord('r') or _k == ord('R'):
        game.reset()
    if _k == ord('1'):
        game.state = STATE_CASUAL ; game.reset()
    if _k == ord('2'):
        game.state = STATE_SPORTS ; game.reset()

cap.release()
cv2.destroyAllWindows()