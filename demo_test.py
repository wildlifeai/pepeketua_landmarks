from landmarks import show_labels, find_landmarks, rotate
import cv2

def main():
    image_path = r"D:\Archeys_frogs\whareorino_b\Grid B\Session 6\13 Feb 2008\IMG_3960.JPG"
    img = cv2.imread(image_path)
    image_df, rotation_prediction = rotate(image_path)
    # show rot
    pred_df = find_landmarks(image_df)
    show_labels(img, pred_df)


if __name__ == '__main__':
    main()