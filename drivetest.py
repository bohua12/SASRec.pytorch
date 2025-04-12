from google.colab import drive
drive.mount('/content/drive')
import shutil
import os

with open("testdrive.txt", "w") as f:
    f.write("Hello test 1")
    f.write("Hello test 2")

drive_path = "/content/drive/collabfiles"
os.makedirs(drive_path, exist_ok=True)
shutil.move("testdrive.txt", os.path.join(drive_path, "testdrive.txt"))
print("File saved to Google Drive.")