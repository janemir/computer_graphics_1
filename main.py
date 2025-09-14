import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches

class Creeper:
    def __init__(self):
        # Матрица координат точек (x, y, z=1)
        self.points = np.array([
            [-4, 16, 1], [4, 16, 1], [4, 8, 1], [3, 8, 1], [-3, 8, 1], [-4, 8, 1],
            [-3, 7, 1], [-4, 7, 1], [-4, -7, 1], [-7, -7, 1], [-7, -16, 1], [0, -16, 1],
            [7, -16, 1], [7, -7, 1], [4, -7, 1], [0, -7, 1], [4, 7, 1], [3, 7, 1],
            [2, 9, 1], [2, 12, 1], [1, 12, 1], [1, 13, 1], [3, 13, 1], [3, 15, 1],
            [1, 15, 1], [-1, 13, 1], [-1, 15, 1], [-3, 15, 1], [-3, 13, 1], [-1, 12, 1],
            [-2, 12, 1], [-2, 9, 1], [-1, 9, 1], [-1, 10, 1], [1, 10, 1], [1, 9, 1]
        ])
        
        # Матрица смежности (индексы начинаются с 0)
        self.adjacency = {
            0: [1, 5],    # 1: 2,6
            1: [0, 2],    # 2: 1,3
            2: [1, 3],    # 3: 2,4
            3: [2, 17],   # 4: 3,18
            4: [3, 5],    # 5: 4,6
            5: [0, 4],    # 6: 1,5
            6: [4, 7, 17],    # 7: 5,8,18
            7: [6, 8],    # 8: 7,9
            8: [7, 9, 15],# 9: 8,10,16
            9: [8, 10],   # 10: 9,11
            10: [9, 11],  # 11: 10,12
            11: [10, 12], # 12: 11,13
            12: [11, 13], # 13: 12,14
            13: [12, 14], # 14: 13,15
            14: [13, 15, 16], # 15: 14,16,17
            15: [8, 11, 14],  # 16: 9,12,15
            16: [14, 17], # 17: 15,18
            17: [3, 16],  # 18: 4,17
            18: [19, 35], # 19: 20,36
            19: [18, 20], # 20: 19,21
            20: [19, 21], # 21: 20,22
            21: [20, 22, 24, 25], # 22: 21,23,25,26
            22: [21, 23], # 23: 22,24
            23: [22, 24], # 24: 23,25
            24: [21, 23], # 25: 22,24
            25: [21, 26, 28, 29], # 26: 22,27,29,30
            26: [25, 27], # 27: 26,28
            27: [26, 28], # 28: 27,29
            28: [25, 27], # 29: 26,28
            29: [25, 30], # 30: 26,31
            30: [29, 31], # 31: 30,32
            31: [30, 32], # 32: 31,33
            32: [31, 33], # 33: 32,34
            33: [32, 34], # 34: 33,35
            34: [33, 35], # 35: 34,36
            35: [18, 34]  # 36: 19,35
        }
        
    def draw(self):
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.set_aspect('equal')
        
        # Устанавливаем розовый фон
        fig.patch.set_facecolor('pink')
        ax.set_facecolor('lavenderblush')  # Очень светлый розовый фон для области графика
        
        # Меняем цвет контура графика (осей) на розовый
        for spine in ax.spines.values():
            spine.set_color('deeppink')
            spine.set_linewidth(2)

        # Рисуем мировую систему координат розовым цветом
        ax.axhline(y=0, color='palevioletred', linestyle='--', alpha=0.7)
        ax.axvline(x=0, color='palevioletred', linestyle='--', alpha=0.7)
        ax.grid(True, alpha=0.3, color='pink')
        
        # Рисуем все связи розовым цветом
        for point_idx, connected_points in self.adjacency.items():
            x1, y1, _ = self.points[point_idx]
            for connected_idx in connected_points:
                x2, y2, _ = self.points[connected_idx]
                ax.plot([x1, x2], [y1, y2], color='deeppink', linewidth=2.5)  # Глубокий розовый
        
        # Рисуем все точки розовым цветом
        x_coords = self.points[:, 0]
        y_coords = self.points[:, 1]
        ax.plot(x_coords, y_coords, 'o', color='hotpink', markersize=6)  # Яркий розовый
        
        # Подписываем точки розово-фиолетовым цветом
        for i, (x, y, _) in enumerate(self.points):
            ax.text(x + 0.2, y + 0.2, str(i+1), fontsize=8, color='mediumvioletred')
        
        # Настройки графика
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        # ax.set_title('Creeper - ЛР №1', fontsize=16, color='mediumvioletred')
        # ax.set_xlabel('Ось X', color='mediumvioletred')
        # ax.set_ylabel('Ось Y', color='mediumvioletred')
        
        # Изменяем цвет меток на осях
        ax.tick_params(colors='mediumvioletred')
        
        plt.show()
    
    def print_info(self):
        """Метод для вывода информации о точках и связях"""
        print("Координаты точек крипера:")
        for i, (x, y, z) in enumerate(self.points):
            print(f"{i+1}: ({x}, {y}, {z})")
        
        print("\nМатрица смежности крипера:")
        for point_idx, connected_points in self.adjacency.items():
            print(f"{point_idx+1}: {[idx+1 for idx in connected_points]}")

# Создаем и рисуем крипера
creeper = Creeper()
creeper.print_info()
creeper.draw()