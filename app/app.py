"""
Signal, Not Noise — Interactive Explorer
=========================================
Keystone Streamlit app for the dimensionality learning repo.

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import gamma
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits, make_swiss_roll

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Signal, Not Noise",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Style ────────────────────────────────────────────────────────────────────

SIGNAL = "#2563EB"
NOISE = "#EF4444"
ACCENT = "#F59E0B"
SUCCESS = "#10B981"
NEUTRAL = "#6B7280"
BG = "#FAFAFA"

mpl.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": NEUTRAL,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "text.color": "#1F2937",
    "font.size": 11,
    "axes.titleweight": "bold",
})


# ── Sidebar navigation ──────────────────────────────────────────────────────

st.sidebar.title("📉 Signal, Not Noise")
st.sidebar.markdown("Interactive dimensionality explorer")

section = st.sidebar.radio(
    "Module",
    [
        "🏠 Overview",
        "01 · Dimensions as Questions",
        "01 · The Curse",
        "02 · Redundancy",
        "03 · PCA Explorer",
        "04 · Nonlinear Methods",
        "05 · Feature Selection",
        "06 · Learned Compression",
        "07 · Decision Framework",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "*Part of the [signal-not-noise](https://github.com/) learning repo.*"
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTIONS
# ══════════════════════════════════════════════════════════════════════════════


# ── Overview ─────────────────────────────────────────────────────────────────

if section == "🏠 Overview":
    st.title("Signal, Not Noise")
    st.markdown(
        """
        > *"Compression is the strategy of whoever actually understands the problem."*

        This app is the interactive companion to the **signal-not-noise** learning
        repo. Each tab corresponds to a module in the notebook series and lets you
        **play with the key concepts** — adjust parameters, watch things break,
        and build intuition you can't get from reading alone.

        ### How to use this

        Pick a module from the sidebar. Each section has **sliders and controls**
        that let you change the parameters of a demonstration. The goal isn't
        to find the "right" settings — it's to develop a feel for how
        dimensionality affects everything.

        ### Learning path

        | Module | Core Question |
        |--------|---------------|
        | **01 · Dimensions** | What is a dimension? Why do more of them hurt? |
        | **02 · Redundancy** | How much of your data is redundant? |
        | **03 · PCA** | How do you find the directions that matter? |
        | **04 · Nonlinear** | What about curved structure? |
        | **05 · Selection** | When should you just throw dimensions away? |
        | **06 · Compression** | Can a neural network learn what to keep? |
        | **07 · Framework** | Which method for which problem? |
        """
    )


# ── 01: Dimensions as Questions ──────────────────────────────────────────────

elif section == "01 · Dimensions as Questions":
    st.title("01 · Dimensions as Questions")
    st.markdown(
        """
        A dimension is just a question you can ask about something.
        Every answer needs its own axis. Let's see what that looks like
        as we add more questions.
        """
    )

    st.subheader("Patient Data: From 1D to 4D")

    n_patients = st.slider("Number of patients", 20, 500, 100, step=20)
    seed = st.slider("Random seed", 0, 100, 42)
    rng = np.random.default_rng(seed)

    age = rng.normal(45, 15, n_patients).clip(18, 90)
    hr = 0.3 * age + rng.normal(70, 10, n_patients)
    spo2 = rng.normal(97, 2, n_patients).clip(80, 100)
    bp = 0.5 * age + rng.normal(120, 15, n_patients)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**1D: Just age**")
        fig1, ax1 = plt.subplots(figsize=(6, 2))
        ax1.scatter(age, np.zeros_like(age), alpha=0.5, s=20, c=SIGNAL)
        ax1.set_xlabel("Age")
        ax1.set_yticks([])
        ax1.set_title("1 question = 1 axis")
        st.pyplot(fig1)
        plt.close()

    with col2:
        st.markdown("**2D: Age + Heart Rate**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(age, hr, alpha=0.5, s=20, c=SIGNAL)
        ax2.set_xlabel("Age")
        ax2.set_ylabel("Heart Rate")
        ax2.set_title("2 questions = 2 axes")
        st.pyplot(fig2)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**3D: Age + HR + SpO2** (rotate in your mind)")
        fig3 = plt.figure(figsize=(6, 5))
        ax3 = fig3.add_subplot(111, projection="3d")
        ax3.scatter(age, hr, spo2, alpha=0.5, s=15, c=SIGNAL)
        ax3.set_xlabel("Age")
        ax3.set_ylabel("HR")
        ax3.set_zlabel("SpO2")
        ax3.set_title("3 questions = 3 axes")
        st.pyplot(fig3)
        plt.close()

    with col4:
        st.markdown("**4D: Colour encodes blood pressure**")
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        scatter = ax4.scatter(age, hr, c=bp, cmap="RdYlBu_r",
                              alpha=0.6, s=25, edgecolors="white",
                              linewidth=0.3)
        ax4.set_xlabel("Age")
        ax4.set_ylabel("Heart Rate")
        ax4.set_title("4th dimension → colour")
        plt.colorbar(scatter, ax=ax4, label="BP Systolic")
        st.pyplot(fig4)
        plt.close()

    st.info(
        "💡 **Key insight**: Each dimension is just another column in your "
        "spreadsheet. 10,000 dimensions = 10,000 columns. The maths doesn't "
        "care that our eyes stop at 3."
    )


# ── 01: The Curse ────────────────────────────────────────────────────────────

elif section == "01 · The Curse":
    st.title("01 · The Curse of Dimensionality")
    st.markdown(
        """
        As dimensions grow, space becomes pathological. Distances lose meaning,
        volumes vanish, and algorithms quietly break.
        """
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Vanishing Sphere",
        "📏 Distance Collapse",
        "📈 Data Hunger",
        "💥 Breaking k-NN",
    ])

    # ── Vanishing sphere ──
    with tab1:
        st.subheader("The Sphere Vanishes Inside the Cube")
        st.markdown(
            "A sphere inscribed in a cube fills *most* of the space in 2D and 3D. "
            "In high dimensions, it fills essentially nothing."
        )

        max_d = st.slider("Maximum dimensions", 10, 200, 50, key="sphere_d")
        dims = np.arange(1, max_d + 1)
        ratios = [
            (np.pi ** (d / 2) / gamma(d / 2 + 1)) * 0.5 ** d
            for d in dims
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(dims, ratios, color=NOISE, alpha=0.8, width=0.8)
        ax.set_xlabel("Dimensions")
        ax.set_ylabel("Sphere Volume / Cube Volume")
        ax.set_title("Inscribed Hypersphere as Fraction of Hypercube")
        if max_d > 20:
            ax.set_yscale("log")
        st.pyplot(fig)
        plt.close()

        st.warning(
            "⚠️ By dimension 20, the sphere occupies less than 0.01% of the cube. "
            "In high dimensions, almost all the volume is in the *corners*."
        )

    # ── Distance collapse ──
    with tab2:
        st.subheader("All Distances Converge")
        st.markdown(
            "Pick a point. Measure its distance to every other point. "
            "In low dimensions, nearest and farthest are very different. "
            "In high dimensions, they're almost the same."
        )

        n_pts = st.slider("Points", 50, 500, 200, step=50, key="dist_n")
        dims_input = st.multiselect(
            "Dimensions to compare",
            [2, 5, 10, 25, 50, 100, 250, 500, 1000, 5000],
            default=[2, 10, 100, 1000],
        )

        if dims_input:
            rng = np.random.default_rng(42)
            n_cols = min(4, len(dims_input))
            n_rows = (len(dims_input) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(4 * n_cols, 3.5 * n_rows))
            axes = np.atleast_2d(axes)

            for idx, d in enumerate(sorted(dims_input)):
                row, col = divmod(idx, n_cols)
                ax = axes[row, col]

                pts = rng.uniform(0, 1, size=(n_pts, d))
                dists = np.linalg.norm(pts[1:] - pts[0], axis=1)
                dists_norm = dists / np.sqrt(d)

                ax.hist(dists_norm, bins=30, alpha=0.7, color=NOISE,
                        edgecolor="white", linewidth=0.5)
                ax.set_title(f"d = {d}")

                contrast = (dists.max() - dists.min()) / (dists.min() + 1e-10)
                ax.text(0.95, 0.85, f"contrast: {contrast:.1f}",
                        transform=ax.transAxes, fontsize=9, ha="right",
                        color=NEUTRAL)

            for idx in range(len(dims_input), n_rows * n_cols):
                row, col = divmod(idx, n_cols)
                axes[row, col].set_visible(False)

            fig.suptitle("Distance Distributions (normalised by √d)",
                         fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.error(
                "🔴 **When contrast → 0, 'nearest neighbour' is meaningless.** "
                "Any algorithm that relies on distances is compromised."
            )

    # ── Data hunger ──
    with tab3:
        st.subheader("Exponential Data Requirements")
        st.markdown(
            "To maintain the same density of coverage as dimensions increase, "
            "you need exponentially more data points."
        )

        density = st.slider("Points per dimension", 3, 20, 10, key="hunger_density")
        max_hunger_d = st.slider("Max dimensions", 5, 30, 20, key="hunger_d")

        dims = np.arange(1, max_hunger_d + 1)
        points_needed = density ** dims.astype(float)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(dims, points_needed, "o-", color=NOISE, markersize=6)
        ax.axhline(y=1e6, color=ACCENT, linestyle="--", alpha=0.7,
                    label="1 million")
        ax.axhline(y=1e9, color="orange", linestyle="--", alpha=0.7,
                    label="1 billion")
        ax.axhline(y=1e80, color="darkred", linestyle="--", alpha=0.5,
                    label="Atoms in observable universe")
        ax.set_xlabel("Dimensions")
        ax.set_ylabel("Points Needed")
        ax.set_title(f"Data Required for {density} points/dim coverage")
        ax.legend()
        st.pyplot(fig)
        plt.close()

        st.info(
            "💡 You will *never* have enough data to fill high-dimensional space. "
            "This is why dimensionality reduction isn't optional — it's survival."
        )

    # ── Breaking k-NN ──
    with tab4:
        st.subheader("Noise Dimensions Destroy Classifiers")
        st.markdown(
            "A simple 2D classification problem. We add noise dimensions "
            "that carry zero information and watch accuracy degrade."
        )

        n_samples = st.slider("Training samples", 100, 1000, 300,
                               step=50, key="knn_n")
        k = st.slider("k (neighbours)", 1, 15, 5, key="knn_k")

        rng = np.random.default_rng(42)
        X_sig = rng.standard_normal((n_samples, 2))
        y = (X_sig[:, 0] + X_sig[:, 1] > 0).astype(int)

        noise_counts = [0, 5, 10, 20, 50, 100, 200, 500]
        accuracies = []

        progress = st.progress(0)
        for i, n_noise in enumerate(noise_counts):
            if n_noise > 0:
                X = np.hstack([X_sig, rng.standard_normal((n_samples, n_noise))])
            else:
                X = X_sig.copy()

            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X, y, cv=5, scoring="accuracy")
            accuracies.append(scores.mean())
            progress.progress((i + 1) / len(noise_counts))

        progress.empty()

        total_dims = [2 + n for n in noise_counts]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(total_dims, accuracies, "o-", color=NOISE, markersize=8,
                linewidth=2)
        ax.axhline(y=accuracies[0], color=SUCCESS, linestyle="--",
                    alpha=0.7, label=f"Signal-only accuracy: {accuracies[0]:.2f}")
        ax.set_xlabel("Total Dimensions (2 signal + noise)")
        ax.set_ylabel("k-NN Accuracy (5-fold CV)")
        ax.set_title("Adding Noise Dimensions Destroys Classification")
        ax.legend()
        ax.set_ylim(0.4, 1.02)
        st.pyplot(fig)
        plt.close()

        st.error(
            "🔴 **Those extra dimensions aren't neutral — they actively damage "
            "performance.** Every dimension that doesn't carry signal is sand "
            "in the gears."
        )


# ── 02: Redundancy ───────────────────────────────────────────────────────────

elif section == "02 · Redundancy":
    st.title("02 · Redundancy — Most Dimensions Are Lying")
    st.markdown(
        "If 8 of your 10 measurements are correlated echoes of the same "
        "2 underlying factors, you're paying 5x the cost for the same signal."
    )

    st.subheader("Interactive Correlation Explorer")

    intrinsic_d = st.slider("True underlying dimensions", 1, 10, 3,
                             key="red_intrinsic")
    ambient_d = st.slider("Recorded dimensions", 10, 200, 50,
                           key="red_ambient")
    noise_level = st.slider("Noise level", 0.01, 1.0, 0.1, step=0.01,
                             key="red_noise")

    rng = np.random.default_rng(42)
    Z = rng.standard_normal((500, intrinsic_d))
    W = rng.standard_normal((intrinsic_d, ambient_d))
    X = Z @ W + rng.standard_normal((500, ambient_d)) * noise_level

    col1, col2 = st.columns(2)

    with col1:
        corr = np.corrcoef(X.T)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(np.abs(corr), cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_title(f"Correlation matrix ({ambient_d} features)")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Feature")
        plt.colorbar(im, ax=ax, label="|Correlation|")
        st.pyplot(fig)
        plt.close()

    with col2:
        pca = PCA().fit(X)
        explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(range(1, len(cumulative) + 1), cumulative, "o-",
                color=SIGNAL, markersize=3)
        ax.axhline(y=0.95, color=ACCENT, linestyle="--", alpha=0.7,
                    label="95% threshold")

        # Find elbow
        n_95 = np.searchsorted(cumulative, 0.95) + 1
        ax.axvline(x=n_95, color=SUCCESS, linestyle="--", alpha=0.7,
                    label=f"{n_95} components for 95%")

        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.set_title("PCA Explained Variance")
        ax.legend()
        ax.set_ylim(0, 1.05)
        st.pyplot(fig)
        plt.close()

    st.success(
        f"✅ **{ambient_d} recorded dimensions, but only ~{intrinsic_d} matter.** "
        f"PCA needs {n_95} components for 95% of the variance. "
        f"The other {ambient_d - n_95} dimensions are mostly noise."
    )


# ── 03: PCA Explorer ─────────────────────────────────────────────────────────

elif section == "03 · PCA Explorer":
    st.title("03 · PCA — Finding the Directions That Matter")
    st.markdown(
        "PCA finds the 'best shadow' to cast — the direction that preserves "
        "the most spread in your data."
    )

    tab1, tab2 = st.tabs(["🎯 2D Projection", "🔢 Digits Dataset"])

    with tab1:
        st.subheader("Project 2D Data onto 1D")
        st.markdown(
            "Drag the angle slider to change the projection direction. "
            "PCA finds the angle that maximises the spread of the red dots."
        )

        angle = st.slider("Projection angle (degrees)", 0, 180, 45,
                           key="pca_angle")

        rng = np.random.default_rng(42)
        X_2d = rng.standard_normal((150, 2)) @ np.array([[2, 1], [1, 3]])

        theta = np.radians(angle)
        direction = np.array([np.cos(theta), np.sin(theta)])
        projected = X_2d @ direction

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Original with projection line
        ax1.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.4, s=20, c=SIGNAL)

        line_extent = 8
        ax1.plot([-line_extent * direction[0], line_extent * direction[0]],
                 [-line_extent * direction[1], line_extent * direction[1]],
                 "r-", linewidth=2, alpha=0.7, label="Projection line")

        # Show projections as red dots on the line
        proj_pts = np.outer(projected, direction)
        ax1.scatter(proj_pts[:, 0], proj_pts[:, 1], c=NOISE, s=10, alpha=0.5)

        ax1.set_xlim(-8, 8)
        ax1.set_ylim(-8, 8)
        ax1.set_aspect("equal")
        ax1.set_title("2D data with projection line")
        ax1.legend()

        # Projected 1D
        ax2.hist(projected, bins=30, color=NOISE, alpha=0.7, edgecolor="white")
        ax2.set_title(f"Projected variance: {projected.var():.2f}")
        ax2.set_xlabel("Projected value")

        # Show PCA optimal
        pca = PCA(n_components=1).fit(X_2d)
        pca_angle = np.degrees(np.arctan2(pca.components_[0, 1],
                                           pca.components_[0, 0])) % 180
        pca_proj = X_2d @ pca.components_[0]
        ax2.axvline(x=0, color=NEUTRAL, linestyle=":", alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info(
            f"💡 PCA would choose **{pca_angle:.0f}°** — the direction with "
            f"maximum variance ({pca_proj.var():.2f}). "
            f"Your current angle gives variance {projected.var():.2f}."
        )

    with tab2:
        st.subheader("PCA on Handwritten Digits (64D → 2D)")

        n_components = st.slider("PCA components", 2, 30, 2, key="pca_digits")

        digits = load_digits()
        pca = PCA(n_components=n_components).fit(digits.data)
        X_pca = pca.transform(digits.data)

        col1, col2 = st.columns(2)

        with col1:
            if n_components >= 2:
                fig, ax = plt.subplots(figsize=(7, 6))
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                                     c=digits.target, cmap="tab10",
                                     s=10, alpha=0.6)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_title(f"Digits in {n_components}-component PCA space")
                plt.colorbar(scatter, ax=ax, label="Digit")
                st.pyplot(fig)
                plt.close()

        with col2:
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.bar(range(1, n_components + 1),
                   pca.explained_variance_ratio_, color=SIGNAL, alpha=0.7)
            ax.set_xlabel("Component")
            ax.set_ylabel("Variance Explained")
            ax.set_title(
                f"Total explained: {cumulative[-1]:.1%} "
                f"(from 64D to {n_components}D)"
            )
            st.pyplot(fig)
            plt.close()


# ── 04: Nonlinear Methods ────────────────────────────────────────────────────

elif section == "04 · Nonlinear Methods":
    st.title("04 · When the Structure Is Curved")
    st.markdown(
        "PCA assumes the important structure is flat. "
        "When it's not, you need methods that can follow curves."
    )

    st.subheader("Swiss Roll: PCA vs t-SNE")

    n_roll = st.slider("Points", 300, 2000, 1000, step=100, key="roll_n")
    noise_roll = st.slider("Noise", 0.0, 2.0, 0.5, step=0.1, key="roll_noise")
    perplexity = st.slider("t-SNE perplexity", 5, 100, 30, key="roll_perp")

    X_roll, colour = make_swiss_roll(n_samples=n_roll, noise=noise_roll,
                                      random_state=42)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original 3D (Swiss Roll)**")
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X_roll[:, 0], X_roll[:, 1], X_roll[:, 2],
                   c=colour, cmap="Spectral", s=5, alpha=0.6)
        ax.set_title("Original 3D")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**PCA → 2D** (linear, misses the curl)")
        pca_roll = PCA(n_components=2).fit_transform(X_roll)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(pca_roll[:, 0], pca_roll[:, 1], c=colour,
                   cmap="Spectral", s=5, alpha=0.6)
        ax.set_title("PCA (linear)")
        ax.set_xticks([])
        ax.set_yticks([])
        st.pyplot(fig)
        plt.close()

    with col3:
        st.markdown("**t-SNE → 2D** (nonlinear, unrolls it)")
        with st.spinner("Running t-SNE..."):
            tsne_roll = TSNE(n_components=2, perplexity=perplexity,
                             random_state=42, max_iter=500).fit_transform(X_roll)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(tsne_roll[:, 0], tsne_roll[:, 1], c=colour,
                   cmap="Spectral", s=5, alpha=0.6)
        ax.set_title(f"t-SNE (perplexity={perplexity})")
        ax.set_xticks([])
        ax.set_yticks([])
        st.pyplot(fig)
        plt.close()

    st.info(
        "💡 PCA projects the roll flat — nearby colours get mixed up. "
        "t-SNE preserves the *neighbourhood structure*, keeping similar "
        "colours together. But notice: t-SNE distorts global distances."
    )


# ── 05: Feature Selection ────────────────────────────────────────────────────

elif section == "05 · Feature Selection":
    st.title("05 · Sometimes Just Throw Them Away")
    st.markdown(
        "Feature selection keeps the original dimensions that matter and "
        "discards the rest. The advantage: interpretability."
    )

    st.subheader("Mutual Information: Which Features Carry Signal?")

    from sklearn.feature_selection import mutual_info_classif

    d_signal = st.slider("Signal features", 2, 8, 3, key="fs_signal")
    d_noise = st.slider("Noise features", 5, 100, 30, key="fs_noise")

    rng = np.random.default_rng(42)
    n = 500

    # Signal features with class structure
    X_sig = rng.standard_normal((n, d_signal))
    y = (X_sig.sum(axis=1) > 0).astype(int)

    # Noise features
    X_noise = rng.standard_normal((n, d_noise))
    X = np.hstack([X_sig, X_noise])

    names = [f"Signal_{i}" for i in range(d_signal)] + \
            [f"Noise_{i}" for i in range(d_noise)]

    mi_scores = mutual_info_classif(X, y, random_state=42)

    # Sort by MI
    order = np.argsort(mi_scores)[::-1]
    sorted_names = [names[i] for i in order]
    sorted_scores = mi_scores[order]
    sorted_colours = [SUCCESS if "Signal" in n else NEUTRAL for n in sorted_names]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(sorted_scores)), sorted_scores,
                  color=sorted_colours, alpha=0.8)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=90, fontsize=8)
    ax.set_ylabel("Mutual Information with Target")
    ax.set_title("Feature Importance: Signal Features Rise to the Top")

    # Add legend
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color=SUCCESS, label="Signal"),
                 Patch(color=NEUTRAL, label="Noise")],
        loc="upper right",
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.success(
        f"✅ Mutual information correctly identifies the {d_signal} signal features. "
        f"The {d_noise} noise features cluster near zero. "
        f"You can safely discard them and keep the originals — "
        f"no transformation needed, full interpretability preserved."
    )


# ── 06: Learned Compression ─────────────────────────────────────────────────

elif section == "06 · Learned Compression":
    st.title("06 · Learned Compression")
    st.markdown(
        "When linear methods like PCA can't capture your data's structure, "
        "neural networks can learn a nonlinear compression. The key idea: "
        "force data through a **bottleneck** and train the network to reconstruct it."
    )

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        torch = None

    if torch is None:
        st.error(
            "**PyTorch is required for this module.** "
            "Install it with: `pip install torch`"
        )
    else:
        digits = load_digits()
        X_digits = digits.data / 16.0  # Normalise to [0, 1]

        class _Autoencoder(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 128), nn.ReLU(),
                    nn.Linear(128, latent_dim),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 128), nn.ReLU(),
                    nn.Linear(128, input_dim), nn.Sigmoid(),
                )

            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z), z

        @st.cache_resource
        def _train_ae(latent_dim, epochs=300, lr=1e-3):
            data = load_digits().data / 16.0
            model = _Autoencoder(64, latent_dim)
            optimiser = torch.optim.Adam(model.parameters(), lr=lr)
            X_t = torch.tensor(data, dtype=torch.float32)
            for _ in range(epochs):
                recon, _ = model(X_t)
                loss = nn.functional.mse_loss(recon, X_t)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            model.eval()
            return model

        tab1, tab2 = st.tabs(["Bottleneck Reconstruction", "PCA vs Autoencoder"])

        with tab1:
            st.subheader("Force Data Through a Bottleneck")
            st.markdown(
                "64 pixel values to **k** hidden units to 64 pixel values. "
                "The network must learn what to keep and what to discard."
            )

            latent_dim = st.select_slider(
                "Bottleneck size (latent dimensions)",
                options=[2, 5, 10, 20, 32],
                value=10,
                key="ae_latent",
            )

            with st.spinner(f"Training autoencoder (k={latent_dim})..."):
                model = _train_ae(latent_dim)

            X_t = torch.tensor(X_digits, dtype=torch.float32)
            with torch.no_grad():
                recon, latent = model(X_t)
            recon_np = recon.numpy()
            latent_np = latent.numpy()

            # Show 10 originals vs reconstructions
            fig, axes = plt.subplots(2, 10, figsize=(12, 3))
            indices = np.linspace(0, len(X_digits) - 1, 10, dtype=int)
            for i, idx in enumerate(indices):
                axes[0, i].imshow(X_digits[idx].reshape(8, 8),
                                  cmap="gray_r", vmin=0, vmax=1)
                axes[0, i].set_xticks([])
                axes[0, i].set_yticks([])
                axes[1, i].imshow(recon_np[idx].reshape(8, 8),
                                  cmap="gray_r", vmin=0, vmax=1)
                axes[1, i].set_xticks([])
                axes[1, i].set_yticks([])
            axes[0, 0].set_ylabel("Original", fontsize=10)
            axes[1, 0].set_ylabel("Reconstructed", fontsize=10)
            fig.suptitle(
                f"Autoencoder Reconstruction (bottleneck = {latent_dim})",
                fontweight="bold",
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            mse = float(np.mean((X_digits - recon_np) ** 2))
            st.metric("Reconstruction Error (MSE)", f"{mse:.4f}")

            # Latent space visualisation (first 2 dims)
            if latent_dim >= 2:
                st.subheader("Latent Space (first 2 dimensions)")
                fig, ax = plt.subplots(figsize=(7, 6))
                scatter = ax.scatter(
                    latent_np[:, 0], latent_np[:, 1],
                    c=digits.target, cmap="tab10", s=10, alpha=0.6,
                )
                ax.set_xlabel("Latent 1")
                ax.set_ylabel("Latent 2")
                ax.set_title(f"Digits in {latent_dim}D Latent Space")
                plt.colorbar(scatter, ax=ax, label="Digit")
                st.pyplot(fig)
                plt.close()

            st.info(
                "**Smaller bottleneck = more compression.** At k=2, the network "
                "must distill 64 pixel values into just 2 numbers, yet digits "
                "remain recognisable. That is learned compression."
            )

        with tab2:
            st.subheader("PCA vs Autoencoder: Reconstruction Error")
            st.markdown(
                "Both compress 64D to k dimensions. PCA is linear; the autoencoder "
                "can learn nonlinear mappings. When does the extra capacity help?"
            )

            dims_to_test = [2, 5, 10, 20, 32]
            pca_errors = []
            ae_errors = []

            progress = st.progress(0)
            for i, k in enumerate(dims_to_test):
                # PCA reconstruction
                pca = PCA(n_components=k).fit(X_digits)
                X_pca_recon = pca.inverse_transform(pca.transform(X_digits))
                pca_errors.append(float(np.mean((X_digits - X_pca_recon) ** 2)))

                # Autoencoder reconstruction
                ae_model = _train_ae(k)
                X_t = torch.tensor(X_digits, dtype=torch.float32)
                with torch.no_grad():
                    ae_recon, _ = ae_model(X_t)
                ae_errors.append(float(np.mean((X_digits - ae_recon.numpy()) ** 2)))
                progress.progress((i + 1) / len(dims_to_test))

            progress.empty()

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(dims_to_test, pca_errors, "o-", color=SIGNAL, linewidth=2,
                    markersize=8, label="PCA (linear)")
            ax.plot(dims_to_test, ae_errors, "s-", color=NOISE, linewidth=2,
                    markersize=8, label="Autoencoder (nonlinear)")
            ax.set_xlabel("Latent Dimensions (k)")
            ax.set_ylabel("Reconstruction Error (MSE)")
            ax.set_title("Linear vs Nonlinear Compression on Digits")
            ax.legend()
            st.pyplot(fig)
            plt.close()

            st.info(
                "**At small k**, the autoencoder's nonlinear capacity lets it "
                "capture curved structure that PCA misses. **As k grows**, PCA "
                "catches up as there is less nonlinearity left to exploit. "
                "The meta-lesson: try PCA first, escalate only if needed."
            )


# ── 07: Decision Framework ──────────────────────────────────────────────────

elif section == "07 · Decision Framework":
    st.title("07 · Which Method for Which Problem?")
    st.markdown(
        "After learning all the methods, the real skill is knowing "
        "when to use each one."
    )

    st.subheader("Interactive Decision Guide")

    need_interpretable = st.radio(
        "Do you need to interpret the original features?",
        ["Yes", "No", "Not sure"],
        horizontal=True,
    )

    if need_interpretable == "Yes":
        st.markdown("### → **Feature Selection** (Module 05)")
        st.markdown(
            "Use mutual information, Lasso, or tree importance to identify "
            "which original features matter. You keep the real columns."
        )
        method = st.radio(
            "Data type?",
            ["Linear relationships", "Nonlinear relationships", "Mixed"],
            horizontal=True,
            key="fs_method",
        )
        if method == "Linear relationships":
            st.info("📌 **Try Lasso regression** — it zeros out irrelevant features automatically.")
        elif method == "Nonlinear relationships":
            st.info("📌 **Try mutual information** or **tree-based importance**.")
        else:
            st.info("📌 **Start with mutual information**, validate with tree importance.")

    elif need_interpretable == "No":
        data_structure = st.radio(
            "Is your data's structure likely linear or nonlinear?",
            ["Linear / Unknown", "Nonlinear / Curved"],
            horizontal=True,
            key="ds_structure",
        )

        if data_structure == "Linear / Unknown":
            st.markdown("### → **PCA** (Module 03)")
            st.markdown(
                "Always try PCA first. It's fast, deterministic, and interpretable "
                "(in terms of variance explained). If the explained variance curve "
                "has a clean elbow, you're done."
            )
            st.info("📌 **Rule of thumb**: Keep components until cumulative variance > 95%.")

        else:
            goal = st.radio(
                "What's your goal?",
                ["Visualisation (2D/3D)", "Preprocessing for ML", "Generative model"],
                horizontal=True,
                key="nl_goal",
            )
            if goal == "Visualisation (2D/3D)":
                st.markdown("### → **UMAP** (Module 04)")
                st.markdown(
                    "UMAP preserves both local and global structure, is faster than "
                    "t-SNE, and is currently the default choice for nonlinear "
                    "visualisation."
                )
                st.warning(
                    "⚠️ Don't over-interpret cluster sizes or distances. "
                    "These methods distort global geometry."
                )
            elif goal == "Preprocessing for ML":
                st.markdown("### → **Autoencoder** (Module 06)")
                st.markdown(
                    "If you have enough data and the structure is genuinely nonlinear, "
                    "a learned compression can outperform PCA. But try PCA first."
                )
            else:
                st.markdown("### → **VAE** (Module 06)")
                st.markdown(
                    "Variational autoencoders give you a smooth, navigable latent space "
                    "that you can sample from and interpolate through."
                )

    else:
        st.markdown("### → **Start with PCA**")
        st.markdown(
            "When in doubt, PCA first. Look at the explained variance curve. "
            "If it captures >90% in a few components, that's your answer. "
            "If not, consider whether you need interpretability to decide next steps."
        )

    st.markdown("---")
    st.markdown(
        """
        ### The Ng Principle

        > *Brute force is the strategy of whoever has the deepest pockets.
        > Compression is the strategy of whoever actually understands the problem.*

        The goal isn't to find the most powerful method. It's to find the
        right subspace — the handful of dimensions where your signal actually lives —
        and work there. Same hardware. Same budget. A fraction of the friction.
        """
    )


# ── Footer ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built as part of the **signal-not-noise** learning repo. "
    "Feynman technique applied to dimensionality reduction."
)
