from pynput import keyboard
from datetime import datetime

LOG_FILE = "key_log.txt"

def on_press(key):
    try:
        key_data = key.char  # For alphabetic characters
    except AttributeError:
        key_data = str(key)  # For special keys

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] Key Pressed: {key_data}"

    print(log_entry)
    # Save to file
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")

def on_release(key):
    # Press ESC to stop the logger
    if key == keyboard.Key.esc:
        print("\n[Logger Stopped]")
        return False

print("🎮 Game Key Tracker Started (press ESC to stop)")
print("-----------------------------------------------")

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
