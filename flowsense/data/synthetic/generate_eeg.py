import numpy as np
import pandas as pd
import os

# Parameters
SAMPLING_RATE = 250  # Hz
DURATION_PER_STATE_SECONDS = 60  # seconds for each state segment
NUM_CHANNELS = 4
FOCUSED_FREQ = 10  # Hz (alpha)
UNFOCUSED_FREQ = 30  # Hz (beta like, or just different characteristics)
AMPLITUDE = 10  # microvolts for base signal
NOISE_LEVEL_FOCUSED = 1.5 # microvolts for focused state
NOISE_LEVEL_UNFOCUSED = 2.5 # microvolts for unfocused state, slightly more noisy

OUTPUT_DIR = "flowsense/data/raw/"
OUTPUT_FILENAME = "synthetic_eeg_data.csv"

def generate_channel_data(t, base_freq, amp, noise_std, add_spikes=False):
    """Generates data for a single EEG channel."""
    signal = amp * np.sin(2 * np.pi * base_freq * t)
    noise = noise_std * np.random.randn(len(t))

    # Add some occasional spikes for unfocused state to make it more distinct
    if add_spikes:
        num_spikes = np.random.randint(5, 15)
        for _ in range(num_spikes):
            spike_idx = np.random.randint(0, len(t))
            spike_amp = (np.random.rand() - 0.5) * amp * 3 # Spikes can be +/-
            signal[spike_idx] += spike_amp
            # Make spike affect a few samples
            for i in range(1, 3):
                if spike_idx + i < len(t): signal[spike_idx+i] += spike_amp / (i+1)
                if spike_idx - i > 0: signal[spike_idx-i] += spike_amp / (i+1)

    return signal + noise

def generate_state_segment(duration_seconds, sampling_rate, num_channels, freq, amp, noise_std, is_focused_state, start_time=0):
    """Generates a segment of EEG data for a given state."""
    num_samples = int(duration_seconds * sampling_rate)
    t = np.linspace(start_time, start_time + duration_seconds - (1/sampling_rate), num_samples)

    all_channels_data = []
    for _ in range(num_channels):
        channel_data = generate_channel_data(t, freq, amp, noise_std, add_spikes=not is_focused_state)
        all_channels_data.append(channel_data)

    labels = np.ones(num_samples) if is_focused_state else np.zeros(num_samples)

    return t, np.array(all_channels_data), labels

def main():
    """Main function to generate and save synthetic EEG data."""
    # Generate 'focused' state data
    t_focused, data_focused, labels_focused = generate_state_segment(
        DURATION_PER_STATE_SECONDS, SAMPLING_RATE, NUM_CHANNELS,
        FOCUSED_FREQ, AMPLITUDE, NOISE_LEVEL_FOCUSED, is_focused_state=True, start_time=0
    )

    # Generate 'unfocused' state data, ensuring continuous time
    last_time_focused = t_focused[-1] + (1/SAMPLING_RATE)
    t_unfocused, data_unfocused, labels_unfocused = generate_state_segment(
        DURATION_PER_STATE_SECONDS, SAMPLING_RATE, NUM_CHANNELS,
        UNFOCUSED_FREQ, AMPLITUDE, NOISE_LEVEL_UNFOCUSED, is_focused_state=False, start_time=last_time_focused
    )

    # Concatenate time, data, and labels
    timestamps = np.concatenate((t_focused, t_unfocused))
    eeg_data = np.concatenate((data_focused, data_unfocused), axis=1) # Concatenate along time axis
    full_labels = np.concatenate((labels_focused, labels_unfocused))

    # Create DataFrame
    df_data = {'timestamp': timestamps}
    for i in range(NUM_CHANNELS):
        df_data[f'ch{i+1}'] = eeg_data[i, :]
    df_data['label'] = full_labels

    df = pd.DataFrame(df_data)

    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    df.to_csv(output_path, index=False)

    print(f"Synthetic EEG data generated successfully and saved to: {output_path}")
    print(f"Data shape: {df.shape}")
    print(f"Unique labels: {df['label'].unique()}")

if __name__ == "__main__":
    main()
