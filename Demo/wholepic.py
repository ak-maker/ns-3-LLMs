import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 创建一个3x2的网格布局
fig, axs = plt.subplots(3, 2, figsize=(35, 20))

# 问题和图形的标题
titles = [
    "User Case Demo Result",
    "Question1:Explain how to call the Scene component of the ray tracer using the Sionna Python package.",
    "Question 2: I have installed sionna. How to perform raytracing?",
    "Question 3: How to compute the impulse response accordingly?",
    "Question 4: Create a 2D visualization of the coverage map of muniche scene.",
    "3D View of Question 4"
]

# 图像文件路径
image_paths = [
    "E:\\python\\sionna-rag\\Demo\\result.png",
    "E:\\python\\sionna-rag\\Demo\\img1.png",
    "E:\\python\\sionna-rag\\Demo\\img2.png",
    "E:\\python\\sionna-rag\\Demo\\img3.png",
    "E:\\python\\sionna-rag\\Demo\\img4.png",
    "E:\\python\\sionna-rag\\Demo\\img5.png"
]

# 遍历每个子图并绘制数据
for ax, title, img_path in zip(axs.ravel(), titles, image_paths):
    ax.set_title(title, fontsize=22, fontweight='bold')
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.axis('off')  # 如果不需要显示坐标轴

# 调整布局
plt.tight_layout()
plt.subplots_adjust(wspace=0.00, hspace=0.3)  # 调整子图间的间距
plt.savefig("./wh.png")
plt.show()
