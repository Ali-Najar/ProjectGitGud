import pandas as pd
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("model_weights/move/training_log.csv")

# Calculate Moving Average (Window = 50 episodes)
df['Moving_Avg'] = df['Total_Reward'].rolling(window=50).mean()

# Plot
plt.figure(figsize=(6,6))
# 1. Raw Data (Faint)
plt.plot(df['Steps'], df['Total_Reward'], alpha=1, color='lightblue', label='Raw Reward')
# 2. Moving Average (Clear)
plt.plot(df['Steps'], df['Moving_Avg'], color='blue', linewidth=1.5, label='50-Ep Moving Avg')
plt.grid()
plt.xlabel("Total Steps")
plt.ylabel("Episode Reward")
plt.title("Camera Agent Training Progress")
plt.legend()
plt.tight_layout()
plt.show()