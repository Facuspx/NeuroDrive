import subprocess, numpy as np, cv2, time

cmd = [
    'rpicam-vid', '-t', '0',
    '--width', '1920', '--height', '1080',
    '--framerate', '15',
    '--codec', 'yuv420',
    '--nopreview',
    '-o', '-',
]

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
frame_size = 1920 * 1080 * 3 // 2
yuv_alto = 1080 * 3 // 2

print('Capturando a 1920x1080 YUV420...')
ts_inicio = time.time()
n = 0

while time.time() - ts_inicio < 10:
    raw = proc.stdout.read(frame_size)
    if len(raw) != frame_size:
        break
    yuv = np.frombuffer(raw, dtype=np.uint8).reshape((yuv_alto, 1920))
    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    frame_small = cv2.resize(frame, (640, 480))
    n += 1
    fps = n / (time.time() - ts_inicio)
    cv2.putText(frame_small, f'1920x1080->640x480 | FPS:{fps:.1f}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow('Full Sensor', frame_small)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

proc.terminate()
proc.wait()
cv2.destroyAllWindows()
print(f'OK: {n} frames, FPS: {fps:.1f}')