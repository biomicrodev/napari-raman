from typing import Optional

import colorcet as cc
import numpy as np
from napari import Viewer
from napari.layers import Points
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
)
from xarray import Dataset

from napari_raman.utils import hex2float

cmap = [hex2float(c) for c in cc.fire]


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
        self.setLayout(layout)

    def resizeEvent(self, event: QResizeEvent):
        width = self.noPointSelectedLabel.rect().width()
        height = self.noPointSelectedLabel.rect().height()

        size: QSize = self.size()
        x = size.width() / 2 - width / 2
        y = size.height() / 2 - height / 2
        self.noPointSelectedLabel.move(x, y)

        super().resizeEvent(event)


class SpectrumViewer:
    def __init__(self, viewer: Viewer, data: Dataset):
        self._view = SpectrumViewerWidget()

        self.data = data

        self._current_index: Optional[int] = None

        # init points
        coordinates = np.asarray(data["coordinates"].values.tolist())
        coordinates = np.flip(coordinates, axis=1)  # XY to YX

        self.points_layer: Points = viewer.add_points(
            coordinates, name="Raman raster", size=50, edge_color="#888"
        )
        self.points_layer.events.highlight.connect(self.on_select)

        # init spectrum
        n = np.asarray(data["wavenumber"]).shape[0]
        self._view.slider.setRange(0, n - 1)
        self._view.slider.valueChanged.connect(lambda _: self.update_wavenumber())
        self.update_wavenumber()
        self.on_select()

    def update_wavenumber(self):
        # update plot
        index = self._view.slider.value()
        wavenumbers = np.asarray(self.data["wavenumber"])
        wavenumber = wavenumbers[index]

        self._view.label.setText(f"{wavenumber:.1f} cm<sup>-1</sup>")
        self._view.vLine.setPos(wavenumber)

        # update points
        at_wavenumber = self.data.sel(wavenumber=wavenumber)
        intensity = np.copy(np.asarray(at_wavenumber["intensity"]))
        intensity[intensity < 0] = 0
        intensity /= intensity.max()
        intensity *= 255
        intensity = intensity.astype(int)

        colors = [cmap[v] for v in intensity]
        self.points_layer.face_color = colors

    def on_select(self, event: Event = None):
        selected_data = self.points_layer.selected_data
        if len(selected_data) == 0:
            _current_point = None
            return
        else:
            self._view.noPointSelectedLabel.setVisible(False)

        # pick first point
        index = sorted(list(selected_data))[0]
        if index == self._current_index:
            return

        self._current_index = index

        # update graph
        coordinates = self.data["coordinates"][index]
        at_coordinates = self.data.sel(coordinates=coordinates)
        wavenumber = at_coordinates["wavenumber"]
        intensity = at_coordinates["intensity"]

        self._view.plot.setData(wavenumber, intensity)
        self._view.vLine._onMoved()
