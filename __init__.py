bl_info = {
    "name": "Cubemap Converter Addon",
    "author": "Plastered_Crab and Ultikynnys (and the py360convert GitHub)",
    "version": (1, 6, 7),
    "blender": (4, 2, 0),
    "location": "View3D > UI",
    "description": "Converts between cubemap images and equirectangular maps",
    "warning": "",
    "wiki_url": "",
    "category": "3D View",
}

import sys
import os
import bpy
import numpy as np
import scipy
import concurrent.futures
from . import py360convert


# ===========================
# Pure Conversion Functions
# ===========================

def srgb_to_linear(srgb):
    """Convert sRGB values to linear RGB."""
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4
    )

def linear_to_srgb(linear):
    """Convert linear RGB values to sRGB."""
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * (linear ** (1/2.4)) - 0.055
    )

def generate_dice_mask(height, width):
    """Generates a boolean mask for a standard 4x3 dice layout."""
    face_size = height // 3
    mask = np.zeros((height, width), dtype=bool)
    faces_indices = [
        (1, 0),  # Up
        (0, 1), (1, 1), (2, 1), (3, 1),  # Left, Front, Right, Back
        (1, 2)  # Down
    ]
    for col, row in faces_indices:
        mask[row*face_size : (row+1)*face_size, col*face_size : (col+1)*face_size] = True
    return mask

def inpaint_image(img, mask, erosion=0):
    """Fills invalid (False) regions in the image using Nearest Neighbor inpainting."""
    if erosion == 0:
        return img
    
    if erosion > 0:
        print(f"Eroding mask by {erosion} pixels...")
        mask = scipy.ndimage.binary_erosion(mask, iterations=erosion)
    
    if np.all(mask):
        return img
    
    print("Inpainting background regions...")
    distances, indices = scipy.ndimage.distance_transform_edt(~mask, return_distances=True, return_indices=True)
    row_indices = indices[0]
    col_indices = indices[1]
    inpainted_img = img.copy()
    
    if img.ndim == 2:
        inpainted_img = img[row_indices, col_indices]
    else:
        for c in range(img.shape[2]):
            inpainted_img[:, :, c] = img[:, :, c][row_indices, col_indices]
    
    return inpainted_img

def process_equirectangular_to_cubemap_data(equirect_pixels, width, height, channels, is_linear, separate_alpha_channel, erosion=0):
    """Pure data processing for Equirectangular -> Cubemap conversion."""
    try:
        if channels == 4:
            try:
                alpha = equirect_pixels[:, :, 3]
                mask = alpha > 0
                equirect_pixels = inpaint_image(equirect_pixels, mask, erosion=erosion)
            except Exception as e:
                print(f"Warning: Inpainting failed: {e}. Proceeding without inpainting.")
        
        if not is_linear:
            equirect_pixels[:, :, :3] = srgb_to_linear(equirect_pixels[:, :, :3])
        
        if channels == 4:
            alpha_equirect = equirect_pixels[:, :, 3]
            rgb_equirect = equirect_pixels[:, :, :3]
        else:
            alpha_equirect = np.ones((height, width), dtype=np.float32)
            rgb_equirect = equirect_pixels[:, :, :3]
        
        face_w = width // 4
        cube_rgb = py360convert.e2c(rgb_equirect, face_w=face_w, cube_format='dice')
        alpha_equirect_expanded = np.stack([alpha_equirect]*3, axis=-1)
        cube_alpha = py360convert.e2c(alpha_equirect_expanded, face_w=face_w, cube_format='dice')[:, :, 0]
        
        if erosion > 0:
            try:
                dice_mask = generate_dice_mask(cube_rgb.shape[0], cube_rgb.shape[1])
                cube_rgb = inpaint_image(cube_rgb, dice_mask, erosion=erosion)
                cube_alpha = inpaint_image(cube_alpha, dice_mask, erosion=erosion)
            except Exception as e:
                print(f"Warning: Output Inpainting failed: {e}. Proceeding.")
        
        if not is_linear:
            cube_rgb = linear_to_srgb(cube_rgb)
        
        return cube_rgb, cube_alpha, cube_rgb.shape[1], cube_rgb.shape[0]
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

def process_cubemap_to_equirectangular_data(cubemap_pixels, width, height, channels, is_linear, separate_alpha_channel, erosion=0):
    """Pure data processing for Cubemap -> Equirectangular conversion."""
    try:
        try:
            dice_mask = generate_dice_mask(height, width)
            cubemap_pixels = inpaint_image(cubemap_pixels, dice_mask, erosion=erosion)
        except Exception as e:
            print(f"Warning: Inpainting failed: {e}. Proceeding without inpainting.")
        
        if not is_linear:
            cubemap_pixels[:, :, :3] = srgb_to_linear(cubemap_pixels[:, :, :3])
        
        if channels == 4:
            alpha_cubemap = cubemap_pixels[:, :, 3]
            rgb_cubemap = cubemap_pixels[:, :, :3]
        else:
            alpha_cubemap = np.ones((height, width), dtype=np.float32)
            rgb_cubemap = cubemap_pixels[:, :, :3]
        
        equirect_width = width // 4 * 8
        equirect_height = height // 3 * 4
        
        equirect_rgb = py360convert.c2e(rgb_cubemap, h=equirect_height, w=equirect_width, cube_format='dice')
        alpha_cubemap_expanded = np.stack([alpha_cubemap]*3, axis=-1)
        equirect_alpha = py360convert.c2e(alpha_cubemap_expanded, h=equirect_height, w=equirect_width, cube_format='dice')[:, :, 0]
        
        if not is_linear:
            equirect_rgb = linear_to_srgb(equirect_rgb)
        
        return equirect_rgb, equirect_alpha, equirect_width, equirect_height
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e


# ===========================
# Helper Functions
# ===========================

def detect_format_from_ext(ext):
    """Returns (is_linear, output_format) based on file extension."""
    ext = ext.lower()
    if ext in ['.exr', '.hdr']:
        return True, 'OPEN_EXR'
    elif ext in ['.png']:
        return False, 'PNG'
    elif ext in ['.jpg', '.jpeg']:
        return False, 'JPEG'
    elif ext in ['.tiff', '.tif']:
        return False, 'TIFF'
    elif ext == '.bmp':
        return False, 'BMP'
    else:
        return False, 'PNG'

def load_image_data(file_path, validate_cubemap=False):
    """
    Loads image and returns dict with metadata.
    Returns: dict with 'pixels', 'width', 'height', 'channels', 'is_linear', 'output_format', 'ext'
    Raises: ValueError if invalid
    """
    img = bpy.data.images.load(file_path)
    width, height = img.size
    channels = len(img.pixels) // (width * height)
    
    if validate_cubemap:
        face_size = height // 3
        expected_width = face_size * 4
        if width != expected_width:
            bpy.data.images.remove(img)
            raise ValueError(f"Invalid cubemap dimensions {width}x{height}, expected {expected_width}x{height}")
    
    ext = os.path.splitext(file_path)[1].lower()
    is_linear, output_format = detect_format_from_ext(ext)
    img.colorspace_settings.name = 'Non-Color'
    
    pixels = np.array(img.pixels[:]).reshape((height, width, channels)).astype(np.float32)
    pixels = pixels[:, :, :4]  # Ensure RGBA
    
    bpy.data.images.remove(img)
    
    return {
        'pixels': pixels,
        'width': width,
        'height': height,
        'channels': channels,
        'is_linear': is_linear,
        'output_format': output_format,
        'ext': ext,
        'path': file_path
    }

def create_or_replace_image(name, width, height, alpha=True, float_buffer=False):
    """Creates a new image, removing any existing image with the same name first."""
    if name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[name])
    return bpy.data.images.new(name, width=width, height=height, alpha=alpha, float_buffer=float_buffer)

def set_image_in_editor(image):
    """Sets the given image in the first found Image Editor space."""
    for area in bpy.context.screen.areas:
        if area.type == 'IMAGE_EDITOR':
            for space in area.spaces:
                if space.type == 'IMAGE_EDITOR':
                    space.image = image
                    return

def save_conversion_result(file_data, rgb_data, alpha_data, width, height, separate_alpha, suffix, keep_in_blender=False):
    """
    Generic save function for both conversion types.
    suffix: e.g., 'EquiToCube' or 'CubeToEqui'
    Returns: saved filename
    """
    output_format = file_data['output_format']
    is_linear = file_data['is_linear']
    ext = file_data['ext']
    path = file_data['path']
    float_buffer = (output_format == 'OPEN_EXR')
    use_half = float_buffer
    
    dir_name = os.path.dirname(path)
    base_name = os.path.basename(path)
    f_name, _ = os.path.splitext(base_name)
    
    if separate_alpha:
        # Save RGB
        img_name = f"{f_name}_{suffix}_RGB" if keep_in_blender else "Temp_RGB"
        img_rgb = create_or_replace_image(img_name, width, height, alpha=True, float_buffer=float_buffer)
        img_rgb.file_format = output_format
        img_rgb.use_half_precision = use_half
        img_rgb.colorspace_settings.name = 'sRGB' if not is_linear else 'Non-Color'
        img_rgb.pixels = np.dstack((rgb_data, np.ones_like(alpha_data))).flatten().tolist()
        save_path_rgb = os.path.join(dir_name, f"{f_name}_{suffix}_rgb{ext}")
        img_rgb.filepath_raw = save_path_rgb
        img_rgb.save()
        
        if keep_in_blender:
            set_image_in_editor(img_rgb)
        else:
            bpy.data.images.remove(img_rgb)
        
        # Save Alpha
        img_name_alpha = f"{f_name}_{suffix}_Alpha" if keep_in_blender else "Temp_Alpha"
        img_alpha = create_or_replace_image(img_name_alpha, width, height, alpha=True, float_buffer=float_buffer)
        img_alpha.file_format = output_format
        img_alpha.use_half_precision = use_half
        img_alpha.colorspace_settings.name = 'Non-Color'
        img_alpha.pixels = np.dstack((alpha_data, alpha_data, alpha_data, np.ones_like(alpha_data))).flatten().tolist()
        save_path_alpha = os.path.join(dir_name, f"{f_name}_{suffix}_alpha{ext}")
        img_alpha.filepath_raw = save_path_alpha
        img_alpha.save()
        
        if not keep_in_blender:
            bpy.data.images.remove(img_alpha)
        
        return os.path.basename(save_path_rgb)
    else:
        # Save combined RGBA
        img_name = f"{f_name}_{suffix}" if keep_in_blender else "Temp_Comb"
        img = create_or_replace_image(img_name, width, height, alpha=True, float_buffer=float_buffer)
        img.file_format = output_format
        img.use_half_precision = use_half
        img.colorspace_settings.name = 'sRGB' if not is_linear else 'Non-Color'
        img.pixels = np.dstack((rgb_data, alpha_data)).flatten().tolist()
        save_path = os.path.join(dir_name, f"{f_name}_{suffix}{ext}")
        img.filepath_raw = save_path
        img.save()
        
        if keep_in_blender:
            set_image_in_editor(img)
        else:
            bpy.data.images.remove(img)
        
        return os.path.basename(save_path)


# ===========================
# Base Modal Operator
# ===========================

class BaseConversionOperator(bpy.types.Operator):
    """Base class for modal threaded conversion operators."""
    
    _timer = None
    _executor = None
    _future = None
    _state = 'IDLE'
    _current_file_data = None
    
    def get_conversion_function(self):
        """Override in subclass to return the processing function."""
        raise NotImplementedError
    
    def get_suffix(self):
        """Override in subclass to return file suffix (e.g., 'EquiToCube')."""
        raise NotImplementedError
    
    def validate_file(self, file_data):
        """Override if specific validation needed. Return True if valid."""
        return True
    
    def should_keep_in_blender(self):
        """Override to decide if result should stay in Blender."""
        return True
    
    def execute(self, context):
        """Must be overridden/called by subclass."""
        self.separate_alpha_channel = context.scene.separate_alpha_channel
        self.erosion = context.scene.inpainting_erosion
        
        self._state = 'IDLE'
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.scene.is_converting = True
        
        return {'RUNNING_MODAL'}
    
    def modal(self, context, event):
        if event.type == 'ESC':
            self.cancel(context)
            return {'CANCELLED'}
        
        if event.type == 'TIMER':
            if self._state == 'IDLE':
                return self.on_idle_state(context)
            elif self._state == 'COMPUTING':
                return self.on_computing_state(context)
        
        return {'PASS_THROUGH'}
    
    def on_idle_state(self, context):
        """Override in subclass to handle loading next file."""
        raise NotImplementedError
    
    def on_computing_state(self, context):
        """Handle completion of computation."""
        if self._future.done():
            try:
                rgb_data, alpha_data, w, h = self._future.result()
                
                saved_name = save_conversion_result(
                    self._current_file_data,
                    rgb_data, alpha_data, w, h,
                    self.separate_alpha_channel,
                    self.get_suffix(),
                    keep_in_blender=self.should_keep_in_blender()
                )
                self.on_success(saved_name)
            except Exception as e:
                self.on_error(e)
            
            return self.on_file_complete(context)
        
        return {'PASS_THROUGH'}
    
    def start_processing(self, file_path, validate_cubemap=False):
        """Common logic to start processing a file."""
        try:
            file_data = load_image_data(file_path, validate_cubemap=validate_cubemap)
            self._current_file_data = file_data
            
            if not self.validate_file(file_data):
                return False
            
            self._future = self._executor.submit(
                self.get_conversion_function(),
                file_data['pixels'], file_data['width'], file_data['height'],
                file_data['channels'], file_data['is_linear'],
                self.separate_alpha_channel, self.erosion
            )
            self._state = 'COMPUTING'
            return True
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return False
    
    def on_success(self, saved_name):
        """Override to handle successful conversion."""
        pass
    
    def on_error(self, error):
        """Override to handle conversion error."""
        print(f"Error processing: {error}")
        import traceback
        traceback.print_exc()
    
    def on_file_complete(self, context):
        """Override to determine next action after file completion."""
        self._cleanup_finish(context)
        return {'FINISHED'}
    
    def _cleanup_cancel(self, context):
        """Internal cleanup when cancelling."""
        if self._executor:
            self._executor.shutdown(wait=False)
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.window_manager.progress_end()
        context.scene.is_converting = False
    
    def _cleanup_finish(self, context):
        """Internal cleanup when finishing."""
        if self._executor:
            self._executor.shutdown(wait=True)
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.window_manager.progress_end()
        context.scene.is_converting = False
    
    def cancel(self, context):
        """Blender's cancel callback - must return None."""
        self._cleanup_cancel(context)
        self.report({'WARNING'}, "Conversion cancelled")


# ===========================
# Single File Operators
# ===========================

class ConvertCubemapToEquirectangularOperator(BaseConversionOperator):
    bl_idname = "addon.convert_cubemap"
    bl_label = "Convert Cubemap to Equirectangular"
    
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.exr;*.hdr;*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp", options={'HIDDEN'})
    
    def get_conversion_function(self):
        return process_cubemap_to_equirectangular_data
    
    def get_suffix(self):
        return "CubeToEqui"
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        if not self.filepath or not os.path.exists(self.filepath):
            self.report({'ERROR'}, "Invalid File Path")
            return {'CANCELLED'}
        
        super().execute(context)
        context.window_manager.progress_begin(0, 1)
        return {'RUNNING_MODAL'}
    
    def on_idle_state(self, context):
        if self.start_processing(self.filepath, validate_cubemap=True):
            return {'PASS_THROUGH'}
        else:
            self.report({'ERROR'}, "Failed to load file")
            self.cancel(context)
            return {'CANCELLED'}
    
    def on_success(self, saved_name):
        self.report({'INFO'}, f"Converted: {saved_name}")


class ConvertEquirectangularToCubemapOperator(BaseConversionOperator):
    bl_idname = "addon.convert_equirectangular"
    bl_label = "Convert Equirectangular to Cubemap"
    
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.exr;*.hdr;*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp", options={'HIDDEN'})
    
    def get_conversion_function(self):
        return process_equirectangular_to_cubemap_data
    
    def get_suffix(self):
        return "EquiToCube"
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        if not self.filepath or not os.path.exists(self.filepath):
            self.report({'ERROR'}, "Invalid File Path")
            return {'CANCELLED'}
        
        super().execute(context)
        context.window_manager.progress_begin(0, 1)
        return {'RUNNING_MODAL'}
    
    def on_idle_state(self, context):
        if self.start_processing(self.filepath):
            return {'PASS_THROUGH'}
        else:
            self.report({'ERROR'}, "Failed to load file")
            self.cancel(context)
            return {'CANCELLED'}
    
    def on_success(self, saved_name):
        self.report({'INFO'}, f"Converted: {saved_name}")


# ===========================
# Batch Operators
# ===========================

class BaseBatchOperator(BaseConversionOperator):
    """Base class for batch processing."""
    
    directory: bpy.props.StringProperty(subtype="DIR_PATH")
    filter_folder: bpy.props.BoolProperty(default=True, options={'HIDDEN'})
    
    _files_to_process = []
    _total_files = 0
    _success_count = 0
    _fail_count = 0
    
    def should_keep_in_blender(self):
        return False  # Batch doesn't keep in Blender
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        if not self.directory or not os.path.exists(self.directory):
            self.report({'ERROR'}, "Invalid Directory")
            return {'CANCELLED'}
        
        # Find all compatible files
        self._files_to_process = []
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".hdr", ".exr", ".tif", ".tiff", ".bmp")):
                    self._files_to_process.append(os.path.join(root, file))
        
        if not self._files_to_process:
            self.report({'WARNING'}, "No compatible files found.")
            return {'CANCELLED'}
        
        self._total_files = len(self._files_to_process)
        self._success_count = 0
        self._fail_count = 0
        
        super().execute(context)
        context.window_manager.progress_begin(0, self._total_files)
        return {'RUNNING_MODAL'}
    
    def on_idle_state(self, context):
        if self._files_to_process:
            file_path = self._files_to_process.pop(0)
            current_index = self._total_files - len(self._files_to_process)
            context.window_manager.progress_update(current_index)
            
            if self.start_processing(file_path, validate_cubemap=self.requires_cubemap_validation()):
                return {'PASS_THROUGH'}
            else:
                self._fail_count += 1
                self._state = 'IDLE'
                return {'PASS_THROUGH'}
        else:
            return self._finish_batch(context)
    
    def requires_cubemap_validation(self):
        """Override if cubemap validation needed."""
        return False
    
    def on_success(self, saved_name):
        self._success_count += 1
    
    def on_error(self, error):
        super().on_error(error)
        self._fail_count += 1
    
    def on_file_complete(self, context):
        self._state = 'IDLE'
        self._future = None
        self._current_file_data = None
        return {'PASS_THROUGH'}
    
    def on_file_complete(self, context):
        """Override to continue processing batch or finish."""
        self._state = 'IDLE'
        self._future = None
        self._current_file_data = None
        return {'PASS_THROUGH'}  # Continue batch processing
    
    def _finish_batch(self, context):
        """Finish batch with reporting."""
        self._cleanup_finish(context)
        if self._fail_count > 0:
            self.report({'WARNING'}, f"Processed {self._success_count} files. Failed: {self._fail_count}. Check console.")
        else:
            self.report({'INFO'}, f"Successfully converted all {self._success_count} files.")
        return {'FINISHED'}


class ConvertAllCubemapsToEquirectangularOperator(BaseBatchOperator):
    bl_idname = "addon.convert_all_cubemaps"
    bl_label = "Convert All Cubemaps in Directory"
    
    def get_conversion_function(self):
        return process_cubemap_to_equirectangular_data
    
    def get_suffix(self):
        return "CubeToEqui"
    
    def requires_cubemap_validation(self):
        return True


class ConvertAllEquirectangularsToCubemapOperator(BaseBatchOperator):
    bl_idname = "addon.convert_all_equirectangulars"
    bl_label = "Convert All Equirectangulars to Cubemap"
    
    def get_conversion_function(self):
        return process_equirectangular_to_cubemap_data
    
    def get_suffix(self):
        return "EquiToCube"


# ===========================
# UI Panel
# ===========================

class ConverterPanel(bpy.types.Panel):
    bl_label = "Cubemap Tool"
    bl_idname = "MYADDON_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Cubemap Tool'
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        if scene.is_converting:
            layout.label(text="Processing...", icon='TIME')
            layout.enabled = False
        
        layout.prop(scene, "separate_alpha_channel")
        layout.prop(scene, "inpainting_erosion", text="Inpainting Erosion (px)")
        
        layout.separator()
        layout.prop(scene, "conversion_mode", expand=True)
        layout.prop(scene, "batch_mode")
        layout.separator()
        
        if scene.conversion_mode == 'CUBE_TO_EQUI':
            if scene.batch_mode:
                layout.operator("addon.convert_all_cubemaps", text="Select Cubemaps Folder", icon='FILE_FOLDER')
            else:
                layout.operator("addon.convert_cubemap", text="Select Cubemap File", icon='FILE_IMAGE')
        else:
            if scene.batch_mode:
                layout.operator("addon.convert_all_equirectangulars", text="Select Equirectangulars Folder", icon='FILE_FOLDER')
            else:
                layout.operator("addon.convert_equirectangular", text="Select Equirectangular File", icon='FILE_IMAGE')


# ===========================
# Registration
# ===========================

def register():
    bpy.utils.register_class(ConvertCubemapToEquirectangularOperator)
    bpy.utils.register_class(ConvertAllCubemapsToEquirectangularOperator)
    bpy.utils.register_class(ConvertEquirectangularToCubemapOperator)
    bpy.utils.register_class(ConvertAllEquirectangularsToCubemapOperator)
    bpy.utils.register_class(ConverterPanel)
    
    bpy.types.Scene.is_converting = bpy.props.BoolProperty(
        name="Is Converting",
        default=False
    )
    
    bpy.types.Scene.batch_mode = bpy.props.BoolProperty(
        name="Batch Mode",
        description="Toggle between single file and batch conversion",
        default=False
    )
    
    bpy.types.Scene.conversion_mode = bpy.props.EnumProperty(
        name="Conversion Mode",
        description="Select conversion direction",
        items=[
            ('CUBE_TO_EQUI', "Cubemap to Equirectangular", "Convert Cubemap images to Equirectangular"),
            ('EQUI_TO_CUBE', "Equirectangular to Cubemap", "Convert Equirectangular images to Cubemap"),
        ],
        default='CUBE_TO_EQUI'
    )
    
    bpy.types.Scene.separate_alpha_channel = bpy.props.BoolProperty(
        name="Separate Alpha Channel",
        description="Handle alpha channel separately",
        default=False
    )
    
    bpy.types.Scene.inpainting_erosion = bpy.props.IntProperty(
        name="Inpainting Edge Erosion",
        description="Pixels to erode from the face edges before inpainting. Set to 0 to disable inpainting.",
        default=2,
        min=0,
        max=50
    )

def unregister():
    bpy.utils.unregister_class(ConvertCubemapToEquirectangularOperator)
    bpy.utils.unregister_class(ConvertAllCubemapsToEquirectangularOperator)
    bpy.utils.unregister_class(ConvertEquirectangularToCubemapOperator)
    bpy.utils.unregister_class(ConvertAllEquirectangularsToCubemapOperator)
    bpy.utils.unregister_class(ConverterPanel)
    
    del bpy.types.Scene.separate_alpha_channel
    del bpy.types.Scene.inpainting_erosion
    del bpy.types.Scene.is_converting
    del bpy.types.Scene.batch_mode
    del bpy.types.Scene.conversion_mode

if __name__ == "__main__":
    register()