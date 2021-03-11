from typing import Optional, Dict, Hashable, Any

import numpy as np
from napari import Viewer
from napari.layers import Points, Layer
from napari.utils.events import Event
from pyqtgraph import (
    PlotDataItem,
    PlotWidget,
    PlotItem,
    AxisItem,
    InfiniteLine,
    TextItem,
)
from qtpy.QtCore import Qt, QSize
from qtpy.QtGui import QFont, QResizeEvent
from qtpy.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QSlider,
    QLabel,
    QStyle,
    QGridLayout,
    QSplitter,
    QScrollArea,
)
from vispy.color import get_colormap
from xarray import Dataset


class JumpSlider(QSlider):
    def mousePressEvent(self, ev):
        """ Jump to click position """
        self.setValue(
            QStyle.sliderValueFromPosition(
                self.minimum(), self.maximum(), ev.x(), self.width()
            )
        )

    def mouseMoveEvent(self, ev):
        """ Jump to pointer position while moving """
        self.setValue(
            QStyle.sliderValueFromPosition(
                self.minimum(), self.maximum(), ev.x(), self.width()
            )
        )


class WavenumberSlider(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.slider = JumpSlider(Qt.Horizontal)
        self.label = QLabel()
        self.label.setStyleSheet("font-size: 14pt")

        layout = QHBoxLayout()
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.label, stretch=0)
        self.setLayout(layout)


class InspectorLine(InfiniteLine):
    def __init__(self):
        super().__init__(angle=90, movable=False)
        self._labels = []
        self._plot_item = None
        self.sigPositionChanged.connect(self._onMoved)

    def _onMoved(self):
        pixelSize, _ = self.getViewBox().viewPixelSize()
        mouseX = self.value()

        self._removeLabels()
        points = []

        # iterate over the existing curves
        for c in self._plot_item.curves:

            # find the index of the closest point of this curve
            if c.xData is None:
                continue

            adiff = np.abs(c.xData - mouseX)
            idx = np.argmin(adiff)

            # set label side to avoid clipping at edges of viewport
            side = (
                "left"
                if (mouseX >= c.xData.min())
                and (mouseX <= (c.xData.max() + c.xData.min()) / 2)
                else "right"
            )

            # only add a label if the line touches the symbol
            tolerance = 0.5 * max(1, c.opts["symbolSize"]) * pixelSize
            if adiff[idx] < tolerance:
                points.append((c.xData[idx], c.yData[idx], side))

        self._createLabels(points)

    def _createLabels(self, points):
        for x, y, side in points:
            text = f"nu={x:.1f}, I={y:.1f}"
            text_item = TextItem(text=text, anchor=(0, 0) if side == "left" else (1.0))
            text_item.setPos(x, y)
            self._labels.append(text_item)
            self._plot_item.addItem(text_item)

    def _removeLabels(self):
        # remove existing texts
        for item in self._labels:
            self._plot_item.removeItem(item)
        self._labels = []

    def attachToPlotItem(self, plot_item):
        self._plot_item = plot_item
        plot_item.addItem(self, ignoreBounds=True)

    def detach(self, plot_item):
        self._removeLabels()
        self._plot_item.removeItem(self)
        self._plot_item = None


class SpectrumViewerWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # attribute viewer
        _label = QLabel()
        _label.setText("<b>Acquisition Parameters</b>")
        _label.setStyleSheet("font-size: 16pt")
        _label.setAlignment(Qt.AlignCenter)

        self.parametersLayout = QGridLayout()
        _parametersLayout = QVBoxLayout()
        _parametersLayout.addWidget(_label, stretch=0)
        _parametersLayout.addSpacing(5)
        _parametersLayout.addLayout(self.parametersLayout, stretch=0)
        _parametersLayout.addWidget(QWidget(), stretch=1)
        _parametersWidget = QWidget()
        _parametersWidget.setLayout(_parametersLayout)
        _parametersScrollArea = QScrollArea()
        _parametersScrollArea.setWidget(_parametersWidget)
        _parametersScrollArea.setWidgetResizable(True)

        # label for user experience purposes
        self.noPointSelectedLabel = QLabel(self)
        self.noPointSelectedLabel.setMinimumWidth(300)
        self.noPointSelectedLabel.setMinimumHeight(60)
        self.noPointSelectedLabel.setAlignment(Qt.AlignCenter)
        self.noPointSelectedLabel.setVisible(True)
        self.noPointSelectedLabel.setText("No point selected")
        self.noPointSelectedLabel.setStyleSheet("font-size: 24pt; color: gray")

        self.plot = PlotDataItem()
        self.vLine = InspectorLine()

        plotWidget = PlotWidget()
        plotWidget.addItem(self.plot)
        self.vLine.attachToPlotItem(plot_item=plotWidget.getPlotItem())

        labelStyle = {"font-size": "14pt", "color": "#FFF"}

        plotItem: PlotItem = plotWidget.getPlotItem()
        plotItem.setLabel("bottom", "Wavenumber (cm<sup>-1</sup>)", **labelStyle)
        plotItem.setLabel("left", "Counts", **labelStyle)

        font = QFont()
        font.setPixelSize(16)
        bottomAxis: AxisItem = plotItem.getAxis("bottom")
        bottomAxis.setStyle(tickFont=font)
        leftAxis: AxisItem = plotItem.getAxis("left")
        leftAxis.setStyle(tickFont=font)

        self.slider = JumpSlider(Qt.Horizontal)
        self.label = QLabel()
        self.label.setStyleSheet("font-size: 14pt")

        wavenumberLayout = QHBoxLayout()
        wavenumberLayout.addWidget(self.slider, stretch=1)
        wavenumberLayout.addWidget(self.label, stretch=0)

        layout = QVBoxLayout()
        layout.addWidget(plotWidget, stretch=1)
        layout.addLayout(wavenumberLayout, stretch=0)
        layoutWidget = QWidget()
        layoutWidget.setLayout(layout)

        splitter = QSplitter()
        splitter.setOrientation(Qt.Vertical)
        splitter.addWidget(_parametersScrollArea)
        splitter.addWidget(layoutWidget)
        splitterLayout = QVBoxLayout()
        splitterLayout.addWidget(splitter)
        self.setLayout(splitterLayout)

    def resizeEvent(self, event: QResizeEvent):
        width = self.noPointSelectedLabel.rect().width()
        height = self.noPointSelectedLabel.rect().height()

        size: QSize = self.size()
        x = size.width() / 2 - width / 2
        y = size.height() / 2 - height / 2
        self.noPointSelectedLabel.move(x, y)

        super().resizeEvent(event)

    def setAcquisitionParameters(self, dct: Dict[Hashable, Any]):
        layout = self.parametersLayout
        keys = sorted([key for key in dct.keys()])
        for row, key in enumerate(keys):
            keyLabel = QLabel()
            keyLabel.setText("<b>" + str(key) + "</b>")
            valueLabel = QLabel()
            valueLabel.setText(str(dct[key]))

            layout.addWidget(keyLabel, row, 0, Qt.AlignRight)
            layout.addWidget(valueLabel, row, 1, Qt.AlignLeft)
            layout.setRowStretch(row, 0)


class SpectrumViewer:
    def __init__(self, viewer: Viewer, dataset: Dataset, cmap: str = "viridis"):
        self.viewer = viewer
        self._view = SpectrumViewerWidget()
        self._view.setAcquisitionParameters(dataset.attrs)

        self._current_layer: Optional[Layer] = None
        self._current_index: Optional[int] = None

        # init points
        self.dataset = dataset
        coords = np.asarray(dataset["coords"].values.tolist())
        coords = np.flip(coords, axis=1)  # XY to YX

        self.layer: Points = viewer.add_points(
            coords, size=20, edge_color="gray", name="Raman raster"
        )
        self.layer.events.highlight.connect(self.on_select)

        self.cmap = get_colormap(cmap)

        # init spectrum
        self.wavenumbers = dataset["wavenumber"].values.tolist()

        self._view.slider.setRange(0, len(self.wavenumbers) - 1)
        self._view.slider.valueChanged.connect(lambda _: self.update_wavenumber())
        self.update_wavenumber()
        self.on_select()

    def update_wavenumber(self):
        # update plot
        index = self._view.slider.value()
        wavenumber = self.wavenumbers[index]

        self._view.label.setText(f"{wavenumber:.1f} cm<sup>-1</sup>")
        self._view.vLine.setPos(wavenumber)

        # update points
        at_wavenumber = self.dataset.sel(wavenumber=wavenumber)
        intensity = np.copy(np.asarray(at_wavenumber["intensity"]))
        intensity = np.ma.array(intensity, mask=intensity <= 0)
        intensity = np.ma.log(intensity)

        min_val = intensity.min()
        max_val = intensity.max()

        intensity -= min_val
        intensity /= max_val - min_val
        intensity = intensity.filled(np.nan)
        intensity = intensity[..., np.newaxis]

        colors = self.cmap.map(intensity)
        self.layer.face_color = colors

    def on_select(self, event: Event = None):
        selected_data = self.layer.selected_data
        if len(selected_data) == 0:
            print("no point selected")
            return
        else:
            self._view.noPointSelectedLabel.setVisible(False)

        # pick first point
        index = sorted(list(selected_data))[0]
        if not self._current_index != index:
            return

        self._current_index = index

        # update graph
        coords = self.dataset["coords"][index]
        at_coordinates = self.dataset.sel(coords=coords)
        wavenumber = at_coordinates["wavenumber"]
        intensity = at_coordinates["intensity"]

        self._view.plot.setData(wavenumber, intensity)
        self._view.vLine._onMoved()
        print("updated")
