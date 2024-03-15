import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation

with open('vgg_logs.json', 'r') as f:
    vgg_data = json.load(f)

with open('resnet50_logs.json', 'r') as f:
    resnet50_data = json.load(f)

fig_loss, ax_loss = plt.subplots(figsize=(8, 6))

line_vgg_loss, = ax_loss.plot([], [], label='VGG Loss')
line_resnet50_loss, = ax_loss.plot([], [], label='ResNet50 Loss')
line_vgg_val_loss, = ax_loss.plot([], [], label='VGG Validation Loss')
line_resnet50_val_loss, = ax_loss.plot([], [], label='ResNet50 Validation Loss')
lines_loss = [line_vgg_loss, line_resnet50_loss, line_vgg_val_loss, line_resnet50_val_loss]

ax_loss.set_xlim(0, max(len(vgg_data['loss']), len(resnet50_data['loss'])))
ax_loss.set_ylim(0, max(max(vgg_data['loss']), max(vgg_data['val_loss']), max(resnet50_data['loss']), max(resnet50_data['val_loss'])))

ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Loss')
ax_loss.set_title('Loss Comparison')
ax_loss.legend()
ax_loss.grid(True)

def update_loss(frame):
    line_vgg_loss.set_data(list(range(1, frame + 1)), vgg_data['loss'][:frame])
    line_resnet50_loss.set_data(list(range(1, frame + 1)), resnet50_data['loss'][:frame])
    line_vgg_val_loss.set_data(list(range(1, frame + 1)), vgg_data['val_loss'][:frame])
    line_resnet50_val_loss.set_data(list(range(1, frame + 1)), resnet50_data['val_loss'][:frame])
    return lines_loss

ani_loss = animation.FuncAnimation(fig_loss, update_loss, frames=min(len(vgg_data['loss']), len(resnet50_data['loss'])), interval=1000, blit=True)

ani_loss.save('loss_comparison.gif', writer='pillow', fps = 25)

print("Successfully created Loss Comparison GIF")

fig_accuracy, ax_accuracy = plt.subplots(figsize=(8, 6))

line_vgg_accuracy, = ax_accuracy.plot([], [], label='VGG Accuracy')
line_resnet50_accuracy, = ax_accuracy.plot([], [], label='ResNet50 Accuracy')
lines_accuracy = [line_vgg_accuracy, line_resnet50_accuracy]

ax_accuracy.set_xlim(0, max(len(vgg_data['accuracy']), len(resnet50_data['accuracy'])))
ax_accuracy.set_ylim(0, 1)

ax_accuracy.set_xlabel('Epoch')
ax_accuracy.set_ylabel('Accuracy')
ax_accuracy.set_title('Accuracy Comparison')
ax_accuracy.legend()
ax_accuracy.grid(True)

def update_accuracy(frame):
    line_vgg_accuracy.set_data(list(range(1, frame + 1)), vgg_data['accuracy'][:frame])
    line_resnet50_accuracy.set_data(list(range(1, frame + 1)), resnet50_data['accuracy'][:frame])
    return lines_accuracy

ani_accuracy = animation.FuncAnimation(fig_accuracy, update_accuracy, frames=min(len(vgg_data['accuracy']), len(resnet50_data['accuracy'])), interval=1000, blit=True)

ani_accuracy.save('accuracy_comparison.gif', writer='pillow', fps = 25)

print("Successfully created Accuracy Comparison GIF")
