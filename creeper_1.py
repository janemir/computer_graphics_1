# Импортируем библиотеку matplotlib.pyplot для создания и управления графиками (визуализация фигуры).
import matplotlib.pyplot as plt
# Импортируем библиотеку numpy для работы с массивами и математическими операциями (необходима для матриц и векторов).
import numpy as np

# Определяем класс Creeper, который управляет фигурой, преобразованиями и отрисовкой (основной контейнер для всей логики программы).
class Creeper:
    # Метод __init__ — конструктор класса, инициализирует все начальные данные и параметры при создании объекта.
    def __init__(self):
        # self.original_points — массив мировых координат точек фигуры в однородном виде [x, y, 1] (N точек, где N=36); это исходные координаты, которые не меняются, только копируются для преобразований (согласно требованию "Хранение мировых координат").
        self.original_points = np.array([
            [-4, 16, 1], [4, 16, 1], [4, 8, 1], [3, 8, 1], [-3, 8, 1], [-4, 8, 1],
            [-3, 7, 1], [-4, 7, 1], [-4, -7, 1], [-7, -7, 1], [-7, -16, 1], [0, -16, 1],
            [7, -16, 1], [7, -7, 1], [4, -7, 1], [0, -7, 1], [4, 7, 1], [3, 7, 1],
            [2, 9, 1], [2, 12, 1], [1, 12, 1], [1, 13, 1], [3, 13, 1], [3, 15, 1],
            [1, 15, 1], [-1, 13, 1], [-1, 15, 1], [-3, 15, 1], [-3, 13, 1], [-1, 12, 1],
            [-2, 12, 1], [-2, 9, 1], [-1, 9, 1], [-1, 10, 1], [1, 10, 1], [1, 9, 1]
        ], dtype=float)  # dtype=float обеспечивает точность вычислений с плавающей точкой.

        # self.adjacency — матрица смежности (словарь), где ключ — индекс точки, значение — список индексов connected точек; определяет ребра фигуры для отрисовки линий между точками (альтернатива списку ребер, но удобнее для графа).
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

        # Параметры преобразований: self.scale — текущий коэффициент масштаба (1.0 — исходный размер).
        self.scale = 1.0
        # self.translation — вектор перемещения [tx, ty] (изначально [0, 0]).
        self.translation = np.array([0.0, 0.0])
        # self.rotation_angle — текущий угол поворота в градусах (0.0 — без поворота).
        self.rotation_angle = 0.0
        # self.rotation_step — шаг изменения угла поворота при нажатии клавиш (5 градусов).
        self.rotation_step = 5.0
        # self.move_step — шаг перемещения по осям при нажатии стрелок (0.5 единиц).
        self.move_step = 0.5
        # self.scale_step — шаг изменения масштаба при нажатии +/- (0.4).
        self.scale_step = 0.4

        # Графические объекты: self.fig — объект фигуры matplotlib (окно графика), изначально None.
        self.fig = None
        # self.ax — объект осей (координатная плоскость) внутри фигуры, изначально None.
        self.ax = None
        # self.lines — список объектов линий (ребер) на графике для последующей очистки.
        self.lines = []
        # self.points_plot — объект точек на графике для последующей очистки.
        self.points_plot = None
        # self.texts — список текстовых меток (не используется в текущей версии, но зарезервировано).
        self.texts = []
        # self.info_text — объект текста с информацией о параметрах.
        self.info_text = None

        # Фиксированные пределы осей (статическое поле): self.fixed_xlim и self.fixed_ylim — границы X и Y (-20 до 20).
        self.fixed_xlim = (-20, 20)
        self.fixed_ylim = (-20, 20)

        # Параметры для явного преобразования координат: self.screen_center_x и self.screen_center_y — центр экрана (здесь 0,0 для простоты); self.screen_scale — масштаб (1.0, так как мировые и экранные совпадают по шкале).
        self.screen_center_x = 0.0  # x0 — смещение по X для перевода в экранные.
        self.screen_center_y = 0.0  # y0 — смещение по Y для перевода в экранные.
        self.screen_scale_x = 1.0   # dx — масштаб по X (можно изменить для зума экрана).
        self.screen_scale_y = 1.0   # dy — масштаб по Y (можно изменить для зума экрана).

    # Метод get_scale_matrix — возвращает матрицу масштабирования 3x3 по формуле из презентации ("Масштабирование по x и y"): [[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]]; применяется к точкам для изменения размера.
    def get_scale_matrix(self, scale_x, scale_y):
        """Матрица масштабирования по x и y (отдельная функция на основе презентации, позволяет комбинировать с другими матрицами через умножение)."""
        return np.array([
            [scale_x, 0, 0],  # Строка для x: умножение на scale_x.
            [0, scale_y, 0],  # Строка для y: умножение на scale_y.
            [0, 0, 1]         # Строка для однородного компонента: остается 1.
        ])

    # Метод get_rotation_matrix — возвращает матрицу поворота 3x3 по формуле из презентации ("Поворот"): [[cosθ, -sinθ, 0], [sinθ, cosθ, 0], [0, 0, 1]], где θ в радианах; поворачивает фигуру вокруг начала координат.
    def get_rotation_matrix(self, angle_deg):
        """Матрица поворота (отдельная функция; угол в градусах конвертируется в радианы для тригонометрических функций, как в презентации)."""
        angle_rad = np.radians(angle_deg)  # Конвертация градусов в радианы (np.radians — функция numpy).
        return np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],  # Строка для x: cosθ * x - sinθ * y.
            [np.sin(angle_rad), np.cos(angle_rad), 0],   # Строка для y: sinθ * x + cosθ * y.
            [0, 0, 1]                                    # Строка для однородного компонента.
        ])

    # Метод get_translation_matrix — возвращает матрицу перемещения 3x3 по формуле из презентации ("сдвиг"/"Перемещение"): [[1, 0, tx], [0, 1, ty], [0, 0, 1]]; сдвигает фигуру на tx по x и ty по y.
    def get_translation_matrix(self, tx, ty):
        """Матрица переноса (отдельная функция; добавляет tx и ty к координатам, как в презентации)."""
        return np.array([
            [1, 0, tx],  # Строка для x: x + tx.
            [0, 1, ty],  # Строка для y: y + ty.
            [0, 0, 1]    # Строка для однородного компонента.
        ])

    # Метод get_reflection_y_matrix — возвращает матрицу отражения по оси Y по формуле из презентации ("Отражение относительно оси y"): [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]; меняет знак x.
    def get_reflection_y_matrix(self):
        """Матрица отражения относительно оси Y (отдельная функция; x' = -x, y' = y, как в презентации)."""
        return np.array([
            [-1, 0, 0],  # Строка для x: -x.
            [0, 1, 0],   # Строка для y: y.
            [0, 0, 1]    # Строка для однородного компонента.
        ])

    # Метод apply_transformations — применяет все преобразования к исходным точкам; комбинирует матрицы и умножает на каждую точку (основан на "Комбинированные преобразования" в презентации).
    def apply_transformations(self):
        """Применяет преобразования к точкам (комбинирует матрицы через умножение @, затем применяет к мировым координатам)."""
        scale_matrix = self.get_scale_matrix(self.scale, self.scale)  # Получаем матрицу масштаба (одинаковый по x/y).
        rotation_matrix = self.get_rotation_matrix(self.rotation_angle)  # Получаем матрицу поворота.
        translation_matrix = self.get_translation_matrix(self.translation[0], self.translation[1])  # Получаем матрицу перемещения.
        reflection_y_matrix = self.get_reflection_y_matrix()  # Получаем матрицу отражения (можно отключить).

        # Комбинируем матрицы: порядок умножения справа налево — сначала отражение, затем масштаб, поворот, перемещение (как в презентации A*B*C*).
        transformation_matrix = translation_matrix @ rotation_matrix @ scale_matrix @ reflection_y_matrix
        transformed_points = np.zeros_like(self.original_points)  # Создаем пустой массив для результата (того же размера).
        for i in range(len(self.original_points)):  # Для каждой точки...
            transformed_points[i] = transformation_matrix @ self.original_points[i]  # ...умножаем матрицу на вектор точки [x, y, 1].
        return transformed_points  # Возвращаем преобразованные мировые координаты.

    # Метод world_to_screen — отдельная функция для перевода мировых координат в экранные (требование задания); теперь явно с формулой X = x0 + x * dx, Y = y0 + y * dy (где x0, y0 — смещение, dx, dy — масштаб); отбрасывает однородный компонент.
    def world_to_screen(self, points):
        """Переводит мировые координаты в экранные явно по формуле X = x0 + x * dx; Y = y0 + y * dy (смещение и масштаб для имитации экрана; здесь dx=dy=1, x0=y0=0 для простоты; можно округлить np.round для целых пикселей)."""
        world_xy = points[:, :2]  # Берем только x и y из [x, y, 1] (отбрасываем однородный компонент).
        screen_points = np.zeros_like(world_xy)  # Создаем массив для экранных координат.
        screen_points[:, 0] = self.screen_center_x + world_xy[:, 0] * self.screen_scale_x  # X = x0 + x * dx (явная формула для X).
        screen_points[:, 1] = self.screen_center_y + world_xy[:, 1] * self.screen_scale_y  # Y = y0 + y * dy (явная формула для Y).
        # screen_points = np.round(screen_points).astype(int)  # Закомментировано: если нужно округление до целых пикселей (как в задании), раскомментировать.
        return screen_points  # Возвращаем экранные координаты (float для точности).

    # Метод screen_to_world — отдельная функция для обратного перевода экранных координат в мировые явно по формуле x = (X - x0) / dx; y = (Y - y0) / dy (для полноты, хотя не используется в программе).
    def screen_to_world(self, screen_points):
        """Обратный перевод экранных координат в мировые явно по формуле x = (X - x0) / dx; y = (Y - y0) / dy (не используется, но добавлено для симметрии)."""
        world_xy = np.zeros_like(screen_points)  # Создаем массив для мировых координат.
        world_xy[:, 0] = (screen_points[:, 0] - self.screen_center_x) / self.screen_scale_x  # x = (X - x0) / dx.
        world_xy[:, 1] = (screen_points[:, 1] - self.screen_center_y) / self.screen_scale_y  # y = (Y - y0) / dy.
        return world_xy  # Возвращаем мировые координаты (без однородного компонента).

    # Метод update_plot — обновляет график: очищает старое, применяет преобразования, переводит в экранные, рисует ребра и точки.
    def update_plot(self):
        """Обновляет график (очищает, применяет трансформации, рисует заново)."""
        if self.fig is None:  # Если фигура не создана, выходим.
            return

        # Очистка предыдущих объектов: удаляем линии (ребра).
        for line in self.lines:
            if line in self.ax.lines:  # Проверяем, существует ли линия на осях.
                line.remove()  # Удаляем линию.
        self.lines = []  # Очищаем список линий.

        # Удаляем точки, если они нарисованы.
        if self.points_plot is not None and self.points_plot in self.ax.lines:
            self.points_plot.remove()  # Удаляем объект точек.

        # Удаляем текстовые метки (если были).
        for text in self.texts:
            if text in self.ax.texts:
                text.remove()  # Удаляем текст.
        self.texts = []  # Очищаем список.

        # Удаляем информационный текст.
        if self.info_text is not None and self.info_text in self.ax.texts:
            self.info_text.remove()  # Удаляем info.

        # Получаем преобразованные мировые координаты.
        world_points = self.apply_transformations()
        # Переводим в экранные только перед отрисовкой (как в задании), используя явную формулу.
        screen_points = self.world_to_screen(world_points)

        # Отрисовка ребер: проходим по матрице смежности.
        for point_idx, connected_points in self.adjacency.items():  # Для каждой точки и её соседей...
            x1, y1 = screen_points[point_idx]  # Берем экранные координаты текущей точки.
            for connected_idx in connected_points:  # Для каждого соединения...
                x2, y2 = screen_points[connected_idx]  # Берем экранные координаты соседней точки.
                line, = self.ax.plot([x1, x2], [y1, y2], color='deeppink', linewidth=2.5)  # Рисуем линию между ними (цвет deeppink, толщина 2.5).
                self.lines.append(line)  # Добавляем в список для будущей очистки.

        # Отрисовка точек: извлекаем все x и y из экранных координат.
        x_coords = screen_points[:, 0]  # Все экранные x-координаты.
        y_coords = screen_points[:, 1]  # Все экранные y-координаты.
        self.points_plot, = self.ax.plot(x_coords, y_coords, 'o', color='hotpink', markersize=6)  # Рисуем точки как круги ('o'), цвет hotpink, размер 6.

        # Формируем строку с текущими параметрами (для отображения).
        info_text_str = (f'Масштаб: {self.scale:.2f}x\n'  # Масштаб с 2 знаками.
                         f'Поворот: {self.rotation_angle:.1f}°\n'  # Поворот с 1 знаком.
                         f'Позиция: ({self.translation[0]:.2f}, {self.translation[1]:.2f})')  # Позиция.
        # Добавляем текст в верхний левый угол (координаты относительно осей).
        self.info_text = self.ax.text(0.02, 0.98, info_text_str, transform=self.ax.transAxes, fontsize=10,
                                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='pink', alpha=0.8))

        # Устанавливаем фиксированные границы осей (статическое поле).
        self.ax.set_xlim(self.fixed_xlim)  # X от -20 до 20.
        self.ax.set_ylim(self.fixed_ylim)  # Y от -20 до 20.

        # Перерисовываем canvas (обновляем график на экране).
        self.fig.canvas.draw()

    # Метод on_key_press — обработчик нажатий клавиш; обновляет параметры и вызывает update_plot.
    def on_key_press(self, event):
        """Обработка нажатий клавиш (изменяет параметры преобразований)."""
        if event.key == '=' or event.key == 'add':  # '+' или '=' — увеличение масштаба.
            self.scale += self.scale_step  # Прибавляем шаг (0.4).
        elif event.key == '-' or event.key == 'subtract':  # '-' — уменьшение масштаба.
            self.scale = max(0.1, self.scale - self.scale_step)  # Вычитаем шаг, но не меньше 0.1.
        elif event.key == 'up':  # Стрелка вверх — перемещение по y вверх.
            self.translation[1] += self.move_step  # ty += 0.5.
        elif event.key == 'down':  # Стрелка вниз — по y вниз.
            self.translation[1] -= self.move_step  # ty -= 0.5.
        elif event.key == 'left':  # Стрелка влево — по x влево.
            self.translation[0] -= self.move_step  # tx -= 0.5.
        elif event.key == 'right':  # Стрелка вправо — по x вправо.
            self.translation[0] += self.move_step  # tx += 0.5.
        elif event.key == 'control':  # Ctrl — поворот вправо (увеличение угла).
            self.rotation_angle += self.rotation_step  # Угол += 5 градусов.
        elif event.key == 'shift':  # Shift — поворот влево (уменьшение угла).
            self.rotation_angle -= self.rotation_step  # Угол -= 5 градусов.

        self.update_plot()  # После изменения параметров обновляем график.

    # Метод draw — создает окно, настраивает вид, подключает обработчики и отображает.
    def draw(self):
        """Создает и отображает окно (основной метод запуска визуализации)."""
        plt.ion()  # Включаем интерактивный режим (график обновляется без закрытия).
        # Создаем фигуру (окно) и оси с размером 12x12 дюймов.
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.ax.set_aspect('equal')  # Устанавливаем равный масштаб по x и y (пропорции сохраняются).

        # Минимизируем отступы вокруг осей, чтобы поле занимало почти весь экран.
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)

        # Максимизируем окно для полноэкранного вида.
        fig_manager = plt.get_current_fig_manager()  # Получаем менеджер окна.
        try:
            fig_manager.window.showMaximized()  # Пытаемся максимизировать (для некоторых бэкендов).
        except AttributeError:
            fig_manager.window.state('zoomed')  # Альтернатива для других бэкендов.

        # Настраиваем цвета: фон окна — lightpink.
        self.fig.patch.set_facecolor('lightpink')
        # Фон осей — lavenderblush.
        self.ax.set_facecolor('lavenderblush')
        # Настраиваем рамки осей (spines): цвет deeppink, толщина 2.
        for spine in self.ax.spines.values():
            spine.set_color('deeppink')
            spine.set_linewidth(2)
        # Добавляем осевые линии: горизонтальная (y=0) пунктирная, цвет palevioletred, прозрачность 0.7.
        self.ax.axhline(y=0, color='palevioletred', linestyle='--', alpha=0.7)
        # Вертикальная (x=0) аналогично.
        self.ax.axvline(x=0, color='palevioletred', linestyle='--', alpha=0.7)
        # Включаем сетку: прозрачность 0.3, цвет pink.
        self.ax.grid(True, alpha=0.3, color='pink')
        # Устанавливаем заголовок: "Creeper - ЛР №1", шрифт 14, цвет mediumvioletred, отступ 10.
        self.ax.set_title('Creeper - ЛР №1', fontsize=14, color='mediumvioletred', pad=10)
        # Цвет меток на осях — mediumvioletred.
        self.ax.tick_params(colors='mediumvioletred')

        # Строка с инструкциями по управлению (текстовая подсказка).
        instructions = ("Инструкция:\n"
                       "+ - увеличение масштаба\n"
                       "- - уменьшение масштаба\n"
                       "Стрелки - перемещение\n"
                       "Ctrl - поворот влево\n"
                       "Shift - поворот вправо")
        # Добавляем инструкции в правый нижний угол (координаты относительно осей).
        self.ax.text(0.98, 0.02, instructions, transform=self.ax.transAxes, fontsize=9,
                     verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.8))

        # Устанавливаем фиксированные границы осей.
        self.ax.set_xlim(self.fixed_xlim)
        self.ax.set_ylim(self.fixed_ylim)

        self.update_plot()  # Первоначальная отрисовка.
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)  # Подключаем обработчик клавиш.
        plt.show(block=True)  # Отображаем окно (block=True — программа ждет закрытия окна).

# Проверяем, запущен ли скрипт напрямую (не импортирован как модуль).
if __name__ == "__main__":
    creeper = Creeper()  # Создаем объект класса Creeper.
    creeper.draw()  # Вызываем метод draw для запуска визуализации.