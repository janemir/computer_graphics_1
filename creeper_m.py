import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
import matplotlib.pyplot as plt  # Переносим импорт сюда

class Creeper:
    def __init__(self):
        original_2d_points = np.array([
            [-4, 16, 1], [4, 16, 1], [4, 8, 1], [3, 8, 1], [-3, 8, 1], [-4, 8, 1],
            [-3, 7, 1], [-4, 7, 1], [-4, -7, 1], [-7, -7, 1], [-7, -16, 1], [0, -16, 1],
            [7, -16, 1], [7, -7, 1], [4, -7, 1], [0, -7, 1], [4, 7, 1], [3, 7, 1],
            [2, 9, 1], [2, 12, 1], [1, 12, 1], [1, 13, 1], [3, 13, 1], [3, 15, 1],
            [1, 15, 1], [-1, 13, 1], [-1, 15, 1], [-3, 15, 1], [-3, 13, 1], [-1, 12, 1],
            [-2, 12, 1], [-2, 9, 1], [-1, 9, 1], [-1, 10, 1], [1, 10, 1], [1, 9, 1]
        ], dtype=float)
        front_points = np.hstack((original_2d_points[:, :2], np.zeros((36, 1)), np.ones((36, 1))))
        back_points = np.hstack((original_2d_points[:, :2], np.full((36, 1), 5), np.ones((36, 1))))
        self.original_points = np.vstack((front_points, back_points))
        original_adjacency = {
            0: [1, 5], 1: [0, 2], 2: [1, 3], 3: [2, 17], 4: [3, 5], 5: [0, 4],
            6: [4, 7, 17], 7: [6, 8], 8: [7, 9, 15], 9: [8, 10], 10: [9, 11],
            11: [10, 12], 12: [11, 13], 13: [12, 14], 14: [13, 15, 16], 15: [8, 11, 14],
            16: [14, 17], 17: [3, 16], 18: [19, 35], 19: [18, 20], 20: [19, 21],
            21: [20, 22, 24, 25], 22: [21, 23], 23: [22, 24], 24: [21, 23],
            25: [21, 26, 28, 29], 26: [25, 27], 27: [26, 28], 28: [25, 27],
            29: [25, 30], 30: [29, 31], 31: [30, 32], 32: [31, 33], 33: [32, 34],
            34: [33, 35], 35: [18, 34]
        }
        self.adjacency = {}
        for k, v in original_adjacency.items():
            self.adjacency[k] = list(v)
        offset = 36
        for k, v in original_adjacency.items():
            self.adjacency[k + offset] = [x + offset for x in v]
        for i in range(36):
            self.adjacency[i].append(i + offset)
            self.adjacency[i + offset].append(i)
        self.scale = 1.0
        self.translation = np.array([0.0, 0.0, 0.0])
        self.rotation_angle_x = 0.0
        self.rotation_angle_y = 0.0
        self.rotation_angle_z = 0.0
        self.rotation_step = 5.0
        self.move_step = 0.5
        self.scale_step = 0.4
        self.fig = None
        self.fixed_xlim = (-20, 20)
        self.fixed_ylim = (-20, 20)
        self.fixed_zlim = (-20, 20)

    def get_scale_matrix(self, scale):
        return np.diag([scale, scale, scale, 1])

    def get_rotation_matrix_x(self, angle_deg):
        angle_rad = np.radians(angle_deg)
        return np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad), 0],
            [0, np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 0, 1]
        ])

    def get_rotation_matrix_y(self, angle_deg):
        angle_rad = np.radians(angle_deg)
        return np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
            [0, 0, 0, 1]
        ])

    def get_rotation_matrix_z(self, angle_deg):
        angle_rad = np.radians(angle_deg)
        return np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0, 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def get_translation_matrix(self, tx, ty, tz):
        return np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])

    def get_reflection_y_matrix(self):
        return np.array([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    def apply_transformations(self):
        scale_matrix = self.get_scale_matrix(self.scale)
        rx = self.get_rotation_matrix_x(self.rotation_angle_x)
        ry = self.get_rotation_matrix_y(self.rotation_angle_y)
        rz = self.get_rotation_matrix_z(self.rotation_angle_z)
        rotation_matrix = rz @ ry @ rx
        translation_matrix = self.get_translation_matrix(self.translation[0], self.translation[1], self.translation[2])
        reflection_y_matrix = self.get_reflection_y_matrix()
        transformation_matrix = translation_matrix @ rotation_matrix @ scale_matrix @ reflection_y_matrix
        transformed_points = np.zeros_like(self.original_points)
        for i in range(len(self.original_points)):
            transformed_points[i] = transformation_matrix @ self.original_points[i]
        return transformed_points

    def update_plot(self):
        world_points = self.apply_transformations()
        x_coords = world_points[:, 0]
        y_coords = world_points[:, 1]
        z_coords = world_points[:, 2]
        lines = []
        for point_idx, connected_points in self.adjacency.items():
            x1, y1, z1 = world_points[point_idx, :3]
            for connected_idx in connected_points:
                x2, y2, z2 = world_points[connected_idx, :3]
                lines.append(go.Scatter3d(
                    x=[x1, x2], y=[y1, y2], z=[z1, z2],
                    mode='lines',
                    line=dict(color='deeppink', width=5)
                ))
        points = go.Scatter3d(
            x=x_coords, y=y_coords, z=z_coords,
            mode='markers',
            marker=dict(size=6, color='hotpink')
        )
        layout = go.Layout(
            scene=dict(
                xaxis=dict(range=self.fixed_xlim, title='X', backgroundcolor='lavenderblush', gridcolor='pink', zerolinecolor='palevioletred'),
                yaxis=dict(range=self.fixed_ylim, title='Y', backgroundcolor='lavenderblush', gridcolor='pink', zerolinecolor='palevioletred'),
                zaxis=dict(range=self.fixed_zlim, title='Z', backgroundcolor='lavenderblush', gridcolor='pink', zerolinecolor='palevioletred'),
                bgcolor='lavenderblush',
                aspectmode='cube'
            ),
            title=dict(text='Creeper - ЛР №1', font=dict(color='mediumvioletred', size=14)),
            paper_bgcolor='lightpink',
            showlegend=False,
            margin=dict(l=0, r=0, t=50, b=0),
            annotations=[
                dict(
                    text=(f"Масштаб: {self.scale:.2f}x<br>"
                          f"Поворот X: {self.rotation_angle_x:.1f}°<br>"
                          f"Поворот Y: {self.rotation_angle_y:.1f}°<br>"
                          f"Поворот Z: {self.rotation_angle_z:.1f}°<br>"
                          f"Позиция: ({self.translation[0]:.2f}, {self.translation[1]:.2f}, {self.translation[2]:.2f})<br>"
                          "Инструкция:<br>"
                          "+ - увеличение масштаба<br>"
                          "- - уменьшение масштаба<br>"
                          "Стрелки - перемещение по x y<br>"
                          "q/e - перемещение по z<br>"
                          "r/f - поворот вокруг x<br>"
                          "t/g - поворот вокруг y<br>"
                          "Ctrl - поворот вокруг z вправо<br>"
                          "Shift - поворот вокруг z влево"),
                    x=0.02, y=0.98, xref="paper", yref="paper",
                    showarrow=False, align="left",
                    font=dict(size=10, color='mediumvioletred'),
                    bgcolor='pink', opacity=0.8
                )
            ]
        )
        self.fig = go.Figure(data=lines + [points], layout=layout)
        return self.fig
    def on_key_press(self, event):
        if event.key == '=' or event.key == 'add':
            self.scale += self.scale_step
        elif event.key == '-' or event.key == 'subtract':
            self.scale = max(0.1, self.scale - self.scale_step)
        elif event.key == 'up':
            self.translation[1] += self.move_step
        elif event.key == 'down':
            self.translation[1] -= self.move_step
        elif event.key == 'left':
            self.translation[0] -= self.move_step
        elif event.key == 'right':
            self.translation[0] += self.move_step
        elif event.key == 'q':
            self.translation[2] += self.move_step
        elif event.key == 'e':
            self.translation[2] -= self.move_step
        elif event.key == 'r':
            self.rotation_angle_x += self.rotation_step
        elif event.key == 'f':
            self.rotation_angle_x -= self.rotation_step
        elif event.key == 't':
            self.rotation_angle_y += self.rotation_step
        elif event.key == 'g':
            self.rotation_angle_y -= self.rotation_step
        elif event.key == 'control':
            self.rotation_angle_z += self.rotation_step
        elif event.key == 'shift':
            self.rotation_angle_z -= self.rotation_step
        self.fig = self.update_plot()
        self.fig.show()

    def draw(self):
        pio.renderers.default = 'browser'
        plt.ion()  # Теперь plt доступен благодаря импорту в начале
        self.fig = self.update_plot()
        self.fig.show()
        fig, ax = plt.subplots(figsize=(1, 1))  # Создаём фигуру только для обработки клавиш
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        plt.show(block=True)

if __name__ == "__main__":
    creeper = Creeper()
    creeper.draw()