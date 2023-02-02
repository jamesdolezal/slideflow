import os
import glfw
import numpy as np
import pandas as pd
import imgui
import slideflow as sf
import multiprocessing as mp
from os.path import join, exists
from slideflow import log
from typing import Optional, Tuple, List
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.image as mpl_image

from .gui_utils import imgui_utils, gl_utils, viewer
from .utils import EasyDict


class OpenGLMosaic(sf.mosaic.Mosaic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _initialize_figure(self, figsize, background):
        pass

    def _add_patch(self, loc, size, **kwargs):
        pass

    def _plot_tile_image(self, image, extent, alpha=1):
        return image

    def _finalize_figure(self):
        pass


class MosaicViewer(viewer.Viewer):

    movable = True
    live    = False

    def __init__(self, mosaic, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mosaic = mosaic
        self.mosaic_x_offset = 0
        self.mosaic_y_offset = 0
        self.zoomed = False
        self.size = 0

    @property
    def view_offset(self):
        return (self.mosaic_x_offset, self.mosaic_y_offset)

    def move(self, dx: float, dy: float) -> None:
        if not self.zoomed:
            pass
        else:
            log.debug("Move: dx={}, dy={}".format(dx, dy))
            self.mosaic_x_offset += dx
            self.mosaic_y_offset += dy

    def refresh_view(self, view_params: Optional[EasyDict] = None) -> None:
        pass

    def reset_view(self, max_w: int, max_h: int):
        pass

    def render(self, max_w: int, max_h: int) -> None:
        """Render the mosaic map."""
        if self.size < min(max_w, max_h):
            self.zoomed = False
        if not self.zoomed:
            self.size = min(max_w, max_h)
            if max_w > self.size:
                self.mosaic_x_offset = (max_w - self.size) / 2
                self.mosaic_y_offset = 0
            else:
                self.mosaic_x_offset = 0
                self.mosaic_y_offset = (max_h - self.size) / 2
        self.width = max_w
        self.height = max_h

        image_size = int(self.size / self.mosaic.num_tiles_x)
        imgui.set_next_window_bg_alpha(0)

        # Set the window position by the offset, in points (not pixels)
        imgui.set_next_window_position(self.viz.offset_x, self.viz.offset_y)
        imgui.set_next_window_size(max_w, max_h)
        imgui.begin("Mosaic", flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_FOCUS_ON_APPEARING | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))
        for (x, y), image in self.mosaic.grid_images.items():
            if isinstance(image, np.ndarray):
                self.mosaic.grid_images[(x, y)] = gl_utils.Texture(
                    image=image, bilinear=True, mipmap=False
                )
            gl_id = self.mosaic.grid_images[(x, y)].gl_id
            pos = (
                self.mosaic_x_offset + x * image_size,
                self.mosaic_y_offset + self.size - (y * image_size)
            )
            imgui.set_cursor_pos(pos)
            imgui.image(gl_id, image_size, image_size)
        imgui.end()
        if 'message' in self.viz.result:
            del self.viz.result.message

    def zoom(self, cx: int, cy: int, dz: float) -> None:
        log.debug("Zoom at ({}, {}): dz={}".format(cx, cy, dz))
        self.zoomed = True
        self.size /= dz
        self.mosaic_x_offset -= (1./dz - 1) * (cx - self.mosaic_x_offset)
        self.mosaic_y_offset -= (1./dz - 1) * (cy - self.mosaic_y_offset)

    def reload(self, **kwargs):
        pass


#----------------------------------------------------------------------------

class AnnotationCapture:

    def __init__(self, mouse_idx=1):
        self.clicking = False
        self.mouse_idx = mouse_idx
        self.annotation_points = []

    def capture(
        self,
        x_range: Tuple[int, int],
        y_range: Tuple[int, int]
    ) -> Tuple[Optional[List[Tuple[int, int]]], bool]:
        """Captures a mouse annotation in the given range.

        Args:
            x_range (tuple(int, int)): Range of pixels to capture an annotation,
                in the horizontal dimension.
            y_range (tuple(int, int)): Range of pixels to capture an annotation,
                in the horizontal dimension.

        Returns:
            A list of tuple with (x, y) coordinates for the annotation.

            A boolean indicating whether the annotation is finished (True)
            or still being drawn (False).
        """
        min_x, max_x = x_range[0], x_range[1]
        min_y, max_y = y_range[0], y_range[1]
        mouse_down = imgui.is_mouse_down(self.mouse_idx)
        mouse_x, mouse_y = imgui.get_mouse_pos()
        in_range = (max_x >= mouse_x) and (mouse_x >= min_x) and (max_y >= mouse_y) and (mouse_y >= min_y)
        if not mouse_down and not self.clicking:
            return None, False
        elif not mouse_down:
            self.clicking = False
            to_return = self.annotation_points
            self.annotation_points = []
            return to_return, True
        elif not self.clicking and not in_range:
            return None, False
        else:
            self.clicking = True
            adj_x = min(max(mouse_x - min_x, 0), max_x - min_x)
            adj_y = min(max(mouse_y - min_y, 0), max_y - min_y)
            self.annotation_points.append((adj_x, adj_y))
            return self.annotation_points, False


class MosaicWidget:
    def __init__(self, viz, width=800, s=3, debug=False):
        self.viz            = viz
        self.show           = True
        self.content_height = 0
        self.coords         = None
        self.visible        = False
        self.width          = 800
        self.show           = s
        self.dpi            = 300
        self.debug          = debug
        self.slidemap       = None
        self.num_tiles_x    = 50
        self.pool           = None
        self.annotation_capture = AnnotationCapture()
        self._umap_path     = None
        self._plot_images   = None
        self._plot_transforms = None
        self._bottom_left   = None
        self._top_right     = None
        self._umap_width    = None
        self._late_render_annotations = []


    def _plot_coords(self):
        # Create figure
        x, y = self.coords[:, 0], self.coords[:, 1]
        fig = plt.figure(figsize=(self.width/self.dpi, self.width/self.dpi), dpi=self.dpi)
        plt.scatter(x, y, s=0.2)
        plt.axis('off')
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        plt.tight_layout()
        ax = plt.gca()

        # Convert figure as texture
        stream = BytesIO()
        plt.savefig(stream, format='raw', transparent=True, dpi=self.dpi)
        pilImage = Image.frombytes('RGBA', size=(self.width, self.width), data=stream.getvalue())
        image_array = mpl_image.pil_to_array(pilImage)
        tex_obj = self._tex_obj = gl_utils.Texture(image=image_array, bilinear=False, mipmap=False)
        self._plot_images = tex_obj.gl_id
        self._plot_transforms = ax.transData.transform

        # Save origin / key information for the images
        self._bottom_left = ax.transData.transform([self.coords[:, 0].min(), self.coords[:, 1].min()])
        self._top_right = ax.transData.transform([self.coords[:, 0].max(), self.coords[:, 1].max()])
        self._umap_width = self._top_right[0] - self._bottom_left[0]

    @property
    def mosaic_kwargs(self):
        return dict(num_tiles_x=self.num_tiles_x, tile_select='first')

    def close(self):
        self.pool.join()
        self.pool.close()
        self.pool = None

    def keyboard_callback(self, key, action):
        if not self.viz._control_down and (key == glfw.KEY_EQUAL and action == glfw.PRESS):
            self.increase_mosaic_resolution()
        if not self.viz._control_down and (key == glfw.KEY_MINUS and action == glfw.PRESS):
            self.decrease_mosaic_resolution()

    def increase_mosaic_resolution(self):
        self.num_tiles_x = int(self.num_tiles_x * 1.5)
        self.mosaic.generate_grid(**self.mosaic_kwargs)
        self.mosaic.plot(pool=self.pool)

    def decrease_mosaic_resolution(self):
        self.num_tiles_x = int(self.num_tiles_x / 1.5)
        self.mosaic.generate_grid(**self.mosaic_kwargs)
        self.mosaic.plot(pool=self.pool)

    # Todo: need preprocessing function for SimCLR
    def load_umap(self, path, model, layers='postconv', subsample=500):
        if path != self._umap_path:
            self._umap_path = path
            if path is not None and exists(join(path, 'encoder')):
                df = pd.read_parquet(join(path, 'slidemap.parquet'))
                self.coords = np.stack((df.x.values, df.y.values), axis=1)
                if subsample and self.coords.shape[0] > subsample:
                    idx = np.random.choice(self.coords.shape[0], subsample)
                    self.coords = self.coords[idx]
                features_model, input_tensor = self.load_model(model, layers=layers)
                self.viz._umap_encoders = self.load_umap_encoder(path, features_model, input_tensor)
                self.viz._async_renderer._umap_encoders = self.viz._umap_encoders
                self._plot_coords()
                self.slidemap = sf.SlideMap(cache=join(self._umap_path, 'slidemap.parquet'))
                log.info(f"Loaded UMAP; displaying {self.coords.shape[0]} points.")

    def generate_mosaic(self, tfrecords):
        if self.pool is None:
            ctx = mp.get_context('fork')
            self.pool = ctx.Pool(os.cpu_count())
        self.mosaic = OpenGLMosaic(self.slidemap, tfrecords=tfrecords, **self.mosaic_kwargs)
        self.mosaic.plot()
        self.viz.set_viewer(MosaicViewer(self.mosaic, **self.viz._viewer_kwargs()))

    def load_model(self, path, layers='postconv', **kwargs):
        is_simclr = sf.util.is_simclr_model_path(path)
        if is_simclr:
            import tensorflow as tf
            from slideflow.simclr import SimCLR
            model = SimCLR(2)
            model.num_features = 128
            model.num_logits = 2
            checkpoint = tf.train.Checkpoint(
                model=model,
                global_step=tf.Variable(0, dtype=tf.int64)
            )
            checkpoint.restore(path).expect_partial()
            inp = tf.keras.layers.InputLayer(input_shape=(96, 96, 3), name='input')
            input_tensor = inp.input
            model.outputs = model(inp.output, training=False)
        else:
            model = sf.model.Features(model, layers=layers, include_logits=True, **kwargs).model
            input_tensor = None
        return model, input_tensor

    def load_umap_encoder(self, path, feature_model, input_tensor=None):
        """Assumes `feature_model` has two outputs: (features, logits)"""
        import tensorflow as tf

        encoder = tf.keras.models.load_model(join(path, 'encoder'))
        encoder._name = f'umap_encoder'
        outputs = [encoder(feature_model.outputs[0])]

        # Add the logits output
        outputs += [feature_model.outputs[-1]]

        # Build the encoder model for all layers
        encoder_model = tf.keras.models.Model(
            inputs=input_tensor if input_tensor is not None else feature_model.input,
            outputs=outputs
        )
        return EasyDict(
            encoder=encoder_model,
            layers=['mosaic_umap'],
            range={'mosaic_umap': np.load(join(path, 'range_clip.npz'))['range']},
            clip={'mosaic_umap': np.load(join(path, 'range_clip.npz'))['clip']}
        )

    def view_menu_options(self):
        if imgui.menu_item('Toggle Mosaic UMAP', enabled=(self.coords is not None))[1]:
            self.show = not self.show

    def late_render(self):
        for _ in range(len(self._late_render_annotations)):
            annotation = self._late_render_annotations.pop()
            gl_utils.draw_roi(annotation, color=1, alpha=1, linewidth=3)

    def render_annotation(self, annotation, origin):
        self._late_render_annotations.append(np.array(annotation) + origin)

    def render(self):
        viz = self.viz

        #self.refresh_umap_path(viz._umap_path)
        viz.args.use_umap_encoders = self.coords is not None

        # --- Draw plot with OpenGL ---------------------------------------

        if self.show and (self.coords is not None):
            imgui.set_next_window_size(self.width, self.width)
            _, self.show = imgui.begin("##layer_plot", closable=True, flags=(imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE))
            tx, ty = imgui.get_window_position()
            draw_list = imgui.get_window_draw_list()
            transform_fn = self._plot_transforms
            draw_list.add_image(self._plot_images, (tx, ty), (tx+self.width, ty+self.width))

            # Draw the bottom-left (red) and top_right (green)
            # coordinate points, to verify alignment
            left_x = self._bottom_left[0]
            right_x = self._top_right[0]
            bottom_y = self.width - self._bottom_left[1]
            top_y = self.width - self._top_right[1]
            self.debug=True
            if self.debug:
                draw_list.add_circle_filled(
                    tx + left_x,
                    ty + bottom_y,
                    self.s,
                    imgui.get_color_u32_rgba(1, 0, 0, 1)
                )
                draw_list.add_circle_filled(
                    tx + right_x,
                    ty + top_y,
                    self.s,
                    imgui.get_color_u32_rgba(0, 1, 0, 1)
                )

            # Draw mosaic map view
            if isinstance(viz.viewer, MosaicViewer) and viz.viewer.zoomed:
                mosaic_ratio = viz.viewer.size / self._umap_width
                _x_off = viz.viewer.mosaic_x_offset / mosaic_ratio
                _y_off = viz.viewer.mosaic_y_offset / mosaic_ratio
                _w = viz.viewer.width / mosaic_ratio
                _h = viz.viewer.height / mosaic_ratio
                draw_list.add_rect(
                    tx + left_x - _x_off,
                    ty + top_y  - _y_off,
                    tx + left_x - _x_off + _w,
                    ty + top_y  - _y_off + _h ,
                    imgui.get_color_u32_rgba(1, 0, 0, 1),
                    thickness=2
                )

            # Plot location of tile
            if 'umap_coords' in viz.result and viz.result.umap_coords:
                fc = viz.result.umap_coords['mosaic_umap']
                transformed = transform_fn(fc)
                draw_list.add_circle_filled(
                    tx + transformed[0],
                    ty + self.width - transformed[1],
                    self.s * 2,
                    imgui.get_color_u32_rgba(1, 0, 0, 1)
                )
            imgui.end()

            # Capture mouse input
            new_annotation, finished = self.annotation_capture.capture(
                x_range=(tx, tx+self.width),
                y_range=(ty, ty+self.width)
            )
            if new_annotation is not None:
                self.render_annotation(new_annotation, origin=(tx, ty))
            if finished:
                print("Captured new annotation!:", new_annotation)

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        return

#----------------------------------------------------------------------------
