import cv2

points = []

# Instructions for which points to click
instructions = [
    "Click: Top-left corner",
    "Click: Top-right corner",
    "Click: Center circle (center of pitch)",
    "Click: Top-left penalty box corner",
    "Click: Top-right penalty box corner",
    "Click: Bottom-left penalty box corner (visible)",
    "Click: Bottom-right penalty box corner (visible)",
    "Click: Left sideline point (around y=1200-1400)",
    "Click: Right sideline point (around y=1200-1400)",
    "Press any key when done"
]

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        idx = len(points)
        print(f"{idx}. [{x}, {y}]  # {instructions[idx-1] if idx-1 < len(instructions) else 'Extra point'}")

img = cv2.imread("input/videos/frame.png")
print("\n=== PITCH CALIBRATION POINT PICKER ===")
print("Click on the following points in order:")
for i, instr in enumerate(instructions[:-1], 1):
    print(f"{i}. {instr}")
print("\nPress any key when done\n")

cv2.imshow("Click points - Follow console instructions", img)
cv2.setMouseCallback("Click points - Follow console instructions", click)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n=== FINAL POINTS (copy to pitch_calibration.json) ===")
print('"image_points_px": [')
for i, point in enumerate(points):
    print(f"    [{point[0]}, {point[1]}]{',' if i < len(points)-1 else ''}")
print("]")