import cv2
import os

def video2imgs(videoPath, imgPath):
    if not os.path.exists(imgPath):
        os.makedirs(imgPath)             # 目標文件夾不存在，則創建
    cap = cv2.VideoCapture(videoPath)    # 獲取影片
    judge = cap.isOpened()               # 判斷是否能打開成功
    print(judge)
    fps = cap.get(cv2.CAP_PROP_FPS)      # 幀率，視頻每秒展示多少張圖片
    print('fps:',fps)

    frames = 1                           # 用於統計所有幀數
    count = 108                         # 用於統計保存的圖片數量
    cut_time = 30                         # 每隔x幀抽一張

    while(judge):
        flag, frame = cap.read()         # 讀取每一張圖片 flag表示是否讀取成功，frame是圖片
        if not flag:
            print(flag)
            print("Process finished!")
            break
        else:
            if frames % cut_time == 0:         # 每隔 cut_time 幀抽一張
                imgname = 'Field_' + str(count).rjust(3,'0') + ".jpg"
                newPath = imgPath + imgname
                # print(imgname)
                cv2.imwrite(newPath, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                # cv2.imencode('.jpg', frame)[1].tofile(newPath)
                count += 1
        frames += 1
    cap.release()
    
video2imgs('./original_data/video/0312_3.mp4','./original_data/test3/')
