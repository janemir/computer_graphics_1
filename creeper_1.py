import matplotlib.pyplot as plt
import numpy as np

class Creeper:
    def __init__(self):

        self.original_points = np.array([
            [-4, 16, 1], [4, 16, 1], [4, 8, 1], [3, 8, 1], [-3, 8, 1], [-4, 8, 1],
            [-3, 7, 1], [-4, 7, 1], [-4, -7, 1], [-7, -7, 1], [-7, -16, 1], [0, -16, 1],
            [7, -16, 1], [7, -7, 1], [4, -7, 1], [0, -7, 1], [4, 7, 1], [3, 7, 1],
            [2, 9, 1], [2, 12, 1], [1, 12, 1], [1, 13, 1], [3, 13, 1], [3, 15, 1],
            [1, 15, 1], [-1, 13, 1], [-1, 15, 1], [-3, 15, 1], [-3, 13, 1], [-1, 12, 1],
            [-2, 12, 1], [-2, 9, 1], [-1, 9, 1], [-1, 10, 1], [1, 10, 1], [1, 9, 1]
        ], dtype=float)  

        self.adjacency = {
            0: [1, 5], 1: [0, 2], 2: [1, 3], 3: [2, 17], 4: [3, 5], 5: [0, 4], 
            6: [4, 7, 17], 7: [6, 8], 8: [7, 9, 15], 9: [8, 10], 10: [9, 11], 
            11: [10, 12], 12: [11, 13], 13: [12, 14], 14: [13, 15, 16], 15: [8, 11, 14], 
            16: [14, 17], 17: [3, 16], 18: [19, 35], 19: [18, 20], 20: [19, 21], 
            21: [20, 22, 24, 25], 22: [21, 23], 23: [22, 24], 24: [21, 23], 
            25: [21, 26, 28, 29], 26: [25, 27], 27: [26, 28], 28: [25, 27], 
            29: [25, 30], 30: [29, 31], 31: [30, 32], 32: [31, 33], 33: [32, 34], 
            34: [33, 35], 35: [18, 34]
        }

        self.scale = 1.0
        self.translation = np.array([0.0, 0.0])
        self.rotation_angle = 0.0
        self.rotation_step = 5.0
        self.move_step = 0.5
        self.scale_step = 0.4
        self.fig = None
        self.ax = None
        self.lines = []
        self.points_plot = None
        self.texts = []
        self.info_text = None

        self.fixed_xlim = (-20, 20)
        self.fixed_ylim = (-20, 20)

        self.screen_center_x = 0.0  
        self.screen_center_y = 0.0  
        self.screen_scale_x = 1.0   
        self.screen_scale_y = 1.0   

    def get_scale_matrix(self, scale_x, scale_y):
        return np.array([
            [scale_x, 0, 0],  
            [0, scale_y, 0],  
            [0, 0, 1]         
        ])

    def get_rotation_matrix(self, angle_deg):
        angle_rad = np.radians(angle_deg)  
        return np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],  
            [np.sin(angle_rad), np.cos(angle_rad), 0],   
            [0, 0, 1]                                   
        ])

    def get_translation_matrix(self, tx, ty):
        return np.array([
            [1, 0, tx],  
            [0, 1, ty],  
            [0, 0, 1]    
        ])

    def apply_transformations(self):
        scale_matrix = self.get_scale_matrix(self.scale, self.scale)  
        rotation_matrix = self.get_rotation_matrix(self.rotation_angle)  
        translation_matrix = self.get_translation_matrix(self.translation[0], self.translation[1])  
     

        transformation_matrix = translation_matrix @ rotation_matrix @ scale_matrix 
        transformed_points = np.zeros_like(self.original_points)  
        for i in range(len(self.original_points)):  
            transformed_points[i] = transformation_matrix @ self.original_points[i]  
        return transformed_points  

    def world_to_screen(self, points):
        world_xy = points[:, :2]  
        screen_points = np.zeros_like(world_xy)  
        screen_points[:, 0] = self.screen_center_x + world_xy[:, 0] * self.screen_scale_x  
        screen_points[:, 1] = self.screen_center_y + world_xy[:, 1] * self.screen_scale_y  
        # screen_points = np.round(screen_points).astype(int)  
        return screen_points  

    def screen_to_world(self, screen_points):
        world_xy = np.zeros_like(screen_points)  
        world_xy[:, 0] = (screen_points[:, 0] - self.screen_center_x) / self.screen_scale_x  
        world_xy[:, 1] = (screen_points[:, 1] - self.screen_center_y) / self.screen_scale_y  
        return world_xy  

    def update_plot(self):
        if self.fig is None:  
            return

        for line in self.lines:
            if line in self.ax.lines:  
                line.remove()  
        self.lines = []  

        if self.points_plot is not None and self.points_plot in self.ax.lines:
            self.points_plot.remove() 

        for text in self.texts:
            if text in self.ax.texts:
                text.remove()  
        self.texts = []  

        if self.info_text is not None and self.info_text in self.ax.texts:
            self.info_text.remove()  

        world_points = self.apply_transformations()
        screen_points = self.world_to_screen(world_points)

        for point_idx, connected_points in self.adjacency.items(): 
            x1, y1 = screen_points[point_idx]  
            for connected_idx in connected_points:  
                x2, y2 = screen_points[connected_idx] 
                line, = self.ax.plot([x1, x2], [y1, y2], color='deeppink', linewidth=2.5) 
                self.lines.append(line)  

        x_coords = screen_points[:, 0]  
        y_coords = screen_points[:, 1]  
        self.points_plot, = self.ax.plot(x_coords, y_coords, 'o', color='hotpink', markersize=6)  

        info_text_str = (f'Масштаб: {self.scale:.2f}x\n'  
                         f'Поворот: {self.rotation_angle:.1f}°\n'  
                         f'Позиция: ({self.translation[0]:.2f}, {self.translation[1]:.2f})')  
        self.info_text = self.ax.text(0.02, 0.98, info_text_str, transform=self.ax.transAxes, fontsize=10,
                                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='pink', alpha=0.8))

        self.ax.set_xlim(self.fixed_xlim)  
        self.ax.set_ylim(self.fixed_ylim)  
        self.fig.canvas.draw()

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
        elif event.key == 'control': 
            self.rotation_angle += self.rotation_step  
        elif event.key == 'shift':  
            self.rotation_angle -= self.rotation_step  

        self.update_plot()  

    def draw(self):
        plt.ion()  
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.ax.set_aspect('equal') 

        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)

        fig_manager = plt.get_current_fig_manager()  
        try:
            fig_manager.window.showMaximized()  
        except AttributeError:
            fig_manager.window.state('zoomed') 

        self.fig.patch.set_facecolor('lightpink')
        self.ax.set_facecolor('lavenderblush')
        for spine in self.ax.spines.values():
            spine.set_color('deeppink')
            spine.set_linewidth(2)

        self.ax.axhline(y=0, color='palevioletred', linestyle='--', alpha=0.7)
        self.ax.axvline(x=0, color='palevioletred', linestyle='--', alpha=0.7)
        self.ax.grid(True, alpha=0.3, color='pink')
        self.ax.set_title('Creeper - ЛР №1', fontsize=14, color='mediumvioletred', pad=10)
        self.ax.tick_params(colors='mediumvioletred')

        instructions = ("Инструкция:\n"
                       "+ - увеличение масштаба\n"
                       "- - уменьшение масштаба\n"
                       "Стрелки - перемещение\n"
                       "Ctrl - поворот влево\n"
                       "Shift - поворот вправо")

        self.ax.text(0.98, 0.02, instructions, transform=self.ax.transAxes, fontsize=9,
                     verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.8))

        self.ax.set_xlim(self.fixed_xlim)
        self.ax.set_ylim(self.fixed_ylim)

        self.update_plot() 
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)  
        plt.show(block=True) 

if __name__ == "__main__":
    creeper = Creeper()  
    creeper.draw()  