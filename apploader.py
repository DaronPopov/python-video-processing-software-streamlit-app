import os
import subprocess
import time
import webbrowser
import threading


def open_browser():
   time.sleep(2)  # Wait for the server to start
   webbrowser.open("http://localhost:8501")


if __name__ == "__main__":
   threading.Thread(target=open_browser).start()
   process = subprocess.Popen(["streamlit", "run", "video_analysis_app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


   # Print the output and errors
   for line in process.stdout:
       print(line.decode(), end='')


   for line in process.stderr:
       print(line.decode(), end='')

