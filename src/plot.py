import matplotlib.pyplot as plt
import numpy as np

closebook_llm = {
    "quality": [18.574, 18.476, 18.23, 18.166],
    "lexical_diversity": [0.4, 0.46, 0.512, 0.582],
    "semantic_diversity": [0.07, 0.09, 0.117, 0.148],

}


rag = {
    "quality": [11.834, 11.652, 11.921, 11.852],
    "lexical_diversity": [0.372, 0.461, 0.486, 0.571],
    "semantic_diversity": [0.059, 0.087, 0.104, 0.129]
}
labels = ["t=0.4", "t=0.7", "t=1.0", "t=1.3"]

def smooth_curve(x, y, num=300):
    x = np.array(x)
    y = np.array(y)

    coeffs = np.polyfit(x, y, deg=2)
    poly = np.poly1d(coeffs)

    x_smooth = np.linspace(x.min(), x.max(), num)
    y_smooth = poly(x_smooth)

    return x_smooth, y_smooth

def plot_metric(ax, x_llm, y_llm, x_rag, y_rag,
                xlabel, ylabel, title, labels):
    x_s, y_s = smooth_curve(x_llm, y_llm)
    ax.plot(x_s, y_s, '--', color='red', label='Close-book LLM')
    for xi, yi, lab in zip(x_llm, y_llm, labels):
        ax.scatter(xi, yi, color='red', marker='s', s=80, edgecolor='black', zorder=5)
        ax.text(xi, yi, lab, fontsize=9, ha='left', va='bottom', color='red')


    x_s, y_s = smooth_curve(x_rag, y_rag)
    ax.plot(x_s, y_s, '--', color='blue', label='RAG')
    for xi, yi, lab in zip(x_rag, y_rag, labels):
        ax.scatter(xi, yi, color='blue', marker='o', s=80, edgecolor='black', zorder=5)
        ax.text(xi, yi, lab, fontsize=9, ha='left', va='bottom', color='blue')

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True)
    ax.legend()

def curve():
    fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharex=False)

    #Quality vs Lexical Diversity
    plot_metric(
        axes[0],
        closebook_llm["lexical_diversity"], closebook_llm["quality"],
        rag["lexical_diversity"], rag["quality"],
        xlabel="Lexical Diversity",
        ylabel="Quality Score",
        title="Close-book LLM & RAG (Quality vs Lexical Diversity)",
        labels=labels,
    )

    # Quality vs Semantic Diversity
    plot_metric(
        axes[1],
        closebook_llm["semantic_diversity"], closebook_llm["quality"],
        rag["semantic_diversity"], rag["quality"],
        xlabel="Semantic Diversity",
        ylabel="Quality Score",
        title="Close-book LLM & RAG (Quality vs Semantic Diversity)",
        labels=labels,
    )

    plt.tight_layout()
    out_path = "./figure/tradeoff_quality_diversity.png"
    plt.savefig(out_path, dpi=300)
    plt.show()

curve()