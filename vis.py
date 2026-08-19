import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_mock_trajectory(seq_len=8, height=64, width=64):
    """
    Generates dummy frame trajectories to simulate ground-truth vs world model visual rollouts.
    """
    real_frames = []
    imagined_frames = []

    for t in range(seq_len):
        # Base image: moving circle (simulating environment dynamics)
        x_pos = int(10 + t * 6)
        y_pos = 32

        real_img = np.zeros((height, width, 3), dtype=np.float32)
        rr, cc = np.ogrid[:height, :width]
        mask = (rr - y_pos) ** 2 + (cc - x_pos) ** 2 <= 6**2
        real_img[mask] = [0.2, 0.8, 0.2]  # Green ball for real env

        # Imagined image: starts sharp, gets progressively blurrier/distorted over time
        imagined_img = real_img.copy()
        if t > 0:
            # Blur & color shift compound over imagination horizon
            noise = np.random.normal(0, 0.05 * t, size=real_img.shape)
            imagined_img = np.clip(imagined_img + noise, 0, 1)

        real_frames.append(real_img)
        imagined_frames.append(imagined_img)

    return np.array(real_frames), np.array(imagined_frames)


def plot_real_vs_imagined_strips(real_frames, imagined_frames):
    """
    Creates a dual-row interactive Plotly frame-strip comparing Real vs. Imagined rollouts side-by-side.

    Args:
        real_frames: np.ndarray of shape (T, H, W, C) in range [0, 1]
        imagined_frames: np.ndarray of shape (T, H, W, C) in range [0, 1]
    """
    seq_len = len(real_frames)

    # Setup subplot grid with 2 rows (Real, Imagined) and seq_len columns
    fig = make_subplots(
        rows=2,
        cols=seq_len,
        subplot_titles=[f"t={t}" for t in range(seq_len)] + [""] * seq_len,
        horizontal_spacing=0.01,
        vertical_spacing=0.08,
    )

    for t in range(seq_len):
        # Top Row: Real Environment Frames
        fig.add_trace(
            go.Image(z=(real_frames[t] * 255).astype(np.uint8)),
            row=1,
            col=t + 1,
        )

        # Bottom Row: Imagined World Model Frames
        fig.add_trace(
            go.Image(z=(imagined_frames[t] * 255).astype(np.uint8)),
            row=2,
            col=t + 1,
        )

        # Hide axes ticks for clean image strip look
        fig.update_xaxes(showticklabels=False, row=1, col=t + 1)
        fig.update_yaxes(showticklabels=False, row=1, col=t + 1)
        fig.update_xaxes(showticklabels=False, row=2, col=t + 1)
        fig.update_yaxes(showticklabels=False, row=2, col=t + 1)

    # Layout styling with explicit row labeling
    fig.update_layout(
        title=dict(
            text="<b>Real vs. Imagined Rollout Comparison</b><br><sup>Top: Real Environment | Bottom: RSSM Imagined Rollout</sup>",
            x=0.5,
            xanchor="center",
        ),
        width=120 * seq_len,
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
        template="plotly_white",
    )

    return fig


# --- Execution Example ---
if __name__ == "__main__":
    real_frames, imagined_frames = generate_mock_trajectory(seq_len=8)
    fig = plot_real_vs_imagined_strips(real_frames, imagined_frames)

    # Display interactively (Browser / Jupyter)
    fig.show()
