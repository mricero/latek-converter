import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_training_metrics(log_dir):
    # 1. Find the event file
    event_file = None
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "tfevents" in file:
                event_file = os.path.join(root, file)
                break
    
    if not event_file:
        print("❌ No event files found!")
        return

    # 2. Load the data
    print(f"📈 Extracting data from: {event_file}")
    ea = EventAccumulator(event_file)
    ea.Reload()

    # Extract 'Loss/Train_Step' (or whatever tag you used in train.py)
    if 'Loss/Train_Step' not in ea.Tags()['scalars']:
        print(f"Available tags: {ea.Tags()['scalars']}")
        return

    data = ea.Scalars('Loss/Train_Step')
    df = pd.DataFrame(data)

    # 3. Create the Plot
    plt.figure(figsize=(10, 6))
    
    # Raw Data (Light color)
    plt.plot(df['step'], df['value'], alpha=0.3, color='#3498db', label='Raw Loss')
    
    # Smoothed Data (Moving average for variability)
    df['smoothed'] = df['value'].rolling(window=20).mean()
    plt.plot(df['step'], df['smoothed'], color='#2980b9', linewidth=2, label='Smoothed Loss')

    plt.title('Training Loss vs. Steps (101M Hybrid OCR)', fontsize=14)
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Save the figure
    plt.savefig('training_progress.png', dpi=300)
    print("✅ Graph saved as training_progress.png")
    plt.show()

if __name__ == "__main__":
    # Point this to your actual log subfolder
    LOG_PATH = "logs/blackwell_ocr_v1" 
    plot_training_metrics(LOG_PATH)