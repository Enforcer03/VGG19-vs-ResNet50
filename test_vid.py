import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
with open('vgg_logs.json', 'r') as f:
    vgg_data = json.load(f)
with open('resnet50_logs.json', 'r') as f:
    resnet50_data = json.load(f)

fig_vgg, ax_vgg = plt.subplots(figsize=(8, 6))
line_vgg_loss, = ax_vgg.plot([], [], label='Loss')
line_vgg_train_accuracy, = ax_vgg.plot([], [], label='Training Accuracy')
line_vgg_val_accuracy, = ax_vgg.plot([], [], label='Validation Accuracy')
lines_vgg = [line_vgg_loss, line_vgg_train_accuracy, line_vgg_val_accuracy]

ax_vgg.set_xlim(0, len(vgg_data['loss']) + 1)
ax_vgg.set_ylim(0, max(max(vgg_data['loss']), max(vgg_data['accuracy']), max(vgg_data['accuracy'])) + 0.1)

ax_vgg.set_xlabel('Epoch')
ax_vgg.set_ylabel('Metrics')
ax_vgg.set_title('VGG Training Metrics')
ax_vgg.legend()
ax_vgg.grid(True)

def update_vgg(frame):
    for line, y_data in zip(lines_vgg, [vgg_data['loss'], vgg_data['accuracy'], vgg_data['accuracy']]):
        line.set_data(list(range(1, frame + 1)), y_data[:frame])
    return lines_vgg

ani_vgg = animation.FuncAnimation(fig_vgg, update_vgg, frames=len(vgg_data['loss']), interval=1000, blit=True)

ani_vgg.save('vgg_training_progress.gif', writer='pillow',fps=20)

print("Successfully created VGG training progress GIF")

fig_resnet50, ax_resnet50 = plt.subplots(figsize=(8, 6))

line_resnet50_loss, = ax_resnet50.plot([], [], label='Loss')
line_resnet50_train_accuracy, = ax_resnet50.plot([], [], label='Training Accuracy')
line_resnet50_val_accuracy, = ax_resnet50.plot([], [], label='Validation Accuracy')
lines_resnet50 = [line_resnet50_loss, line_resnet50_train_accuracy, line_resnet50_val_accuracy]

ax_resnet50.set_xlim(0, len(resnet50_data['loss']) + 1)
ax_resnet50.set_ylim(0, max(max(resnet50_data['loss']), max(resnet50_data['accuracy']), max(resnet50_data['accuracy'])) + 0.1)

ax_resnet50.set_xlabel('Epoch')
ax_resnet50.set_ylabel('Metrics')
ax_resnet50.set_title('ResNet50 Training Metrics')
ax_resnet50.legend()
ax_resnet50.grid(True)

def update_resnet50(frame):
    for line, y_data in zip(lines_resnet50, [resnet50_data['loss'], resnet50_data['accuracy'], resnet50_data['accuracy']]):
        line.set_data(list(range(1, frame + 1)), y_data[:frame])
    return lines_resnet50

ani_resnet50 = animation.FuncAnimation(fig_resnet50, update_resnet50, frames=len(resnet50_data['loss']), interval=1000, blit=True)

ani_resnet50.save('resnet50_training_progress.gif', writer='pillow', fps=20)

print("Successfully created ResNet50 training progress GIF")
