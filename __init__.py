bl_info = {
    "name": "Cubemap Converter Addon",
    "author": "Plastered_Crab and Ultikynnys (and the py360convert GitHub)",
    "version": (1, 6, 6),
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





def srgb_to_linear(srgb):
    """Convert sRGB values to linear RGB."""
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4
    )
    return linear

def linear_to_srgb(linear):
    """Convert linear RGB values to sRGB."""
    srgb = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * (linear ** (1/2.4)) - 0.055
    )
    return srgb

def process_equirectangular_to_cubemap_data(equirect_pixels, width, height, channels, is_linear, separate_alpha_channel, erosion=0):
    """
    Pure data processing for Equirectangular -> Cubemap conversion.
    Returns: (cube_rgb, cube_alpha, width_c, height_c)
    """
    try:
        # Inpaint background to prevent edge seams if alpha exists
        if channels == 4:
            try:
                # Create mask from alpha channel (True = Valid, False = Invalid/Background)
                alpha = equirect_pixels[:, :, 3]
                mask = alpha > 0
                equirect_pixels = inpaint_image(equirect_pixels, mask, erosion=erosion)
            except Exception as e:
                print(f"Warning: Inpainting failed: {e}. Proceeding without inpainting.")

        # Convert sRGB to linear if necessary
        if not is_linear:
            equirect_pixels[:, :, :3] = srgb_to_linear(equirect_pixels[:, :, :3])

        # Separate alpha channel if needed
        if channels == 4:
            alpha_equirect = equirect_pixels[:, :, 3]
            rgb_equirect = equirect_pixels[:, :, :3]
        else:
            alpha_equirect = np.ones((height, width), dtype=np.float32)
            rgb_equirect = equirect_pixels[:, :, :3]

        # Determine face width based on the width of the equirectangular image
        face_w = width // 4

        # Convert RGB equirectangular to cubemap
        cube_rgb = py360convert.e2c(rgb_equirect, face_w=face_w, cube_format='dice')

        # Convert alpha equirectangular to cubemap
        alpha_equirect_expanded = np.stack([alpha_equirect]*3, axis=-1)
        cube_alpha = py360convert.e2c(alpha_equirect_expanded, face_w=face_w, cube_format='dice')[:, :, 0]

        # Post-process: Inpaint the output dice to extend faces into the void (Padding)
        # This prevents edge bleeding when mipmapping.
        if erosion > 0:
            try:
                # Dice mask logic is same for input/output if dimensions match dice spec
                dice_mask = generate_dice_mask(cube_rgb.shape[0], cube_rgb.shape[1])
                cube_rgb = inpaint_image(cube_rgb, dice_mask, erosion=erosion)
                cube_alpha = inpaint_image(cube_alpha, dice_mask, erosion=erosion)
            except Exception as e:
                print(f"Warning: Output Inpainting failed: {e}. Proceeding.")

        # Convert linear to sRGB if saving in sRGB format
        if not is_linear:
            cube_rgb = linear_to_srgb(cube_rgb)

        return cube_rgb, cube_alpha, cube_rgb.shape[1], cube_rgb.shape[0]

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e


def convert_equirectangular_to_cubemap(equirectangular_image_path, separate_alpha_channel):
    print(f"Processing equirectangular image: {equirectangular_image_path}")
    try:
        # Load the equirectangular image
        try:
            equirect_image = bpy.data.images.load(equirectangular_image_path)
        except Exception as e:
            print(f"Failed to load image {equirectangular_image_path}: {e}")
            return False, f"Failed to load: {e}"

        print("Image loaded successfully.")

        # Determine the image color space and format based on file extension
        ext = os.path.splitext(equirectangular_image_path)[1].lower()
        if ext in ['.exr', '.hdr']:
            is_linear = True
            output_format = 'OPEN_EXR'
            equirect_image.colorspace_settings.name = 'Non-Color'
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            is_linear = False
            output_format = ext.replace('.', '').upper()
            if output_format == 'JPG':
                output_format = 'JPEG'
            elif output_format == 'TIF':
                output_format = 'TIFF'
            equirect_image.colorspace_settings.name = 'Non-Color'  # Load without color management
        else:
            print(f"Unsupported input image format: {ext}")
            return False, f"Unsupported format: {ext}"

        width, height = equirect_image.size
        channels = len(equirect_image.pixels) // (width * height)

        print(f"Image size: width={width}, height={height}, channels={channels}")

        # Read the pixels and reshape
        equirect_pixels = np.array(equirect_image.pixels[:]).reshape((height, width, channels)).astype(np.float32)
        equirect_pixels = equirect_pixels[:, :, :4]  # Ensure RGBA

        # --- Delegate Math to Pure Function ---
        cube_rgb, cube_alpha, width_c, height_c = process_equirectangular_to_cubemap_data(
            equirect_pixels, width, height, channels, is_linear, separate_alpha_channel, erosion=erosion
        )
        # --------------------------------------

        # Clamp values between 0 and 1 for 8-bit formats
        if output_format in ['PNG', 'JPEG', 'TIFF', 'BMP']:
            cube_rgb = np.clip(cube_rgb, 0.0, 1.0)

        # Get dimensions from cube_rgb
        height_c, width_c, _ = cube_rgb.shape

        # Determine float_buffer and use_half_precision settings
        if output_format == 'OPEN_EXR':
            float_buffer = True
            use_half_precision = True
        else:
            float_buffer = False
            use_half_precision = False

        if separate_alpha_channel:
            # Create RGB cubemap image with alpha channel set to 1
            cube_rgb_image = create_or_replace_image(
                "Cubemap RGB Image",
                width=width_c,
                height=height_c,
                alpha=True,
                float_buffer=float_buffer
            )
            cube_rgb_image.use_half_precision = use_half_precision
            cube_rgb_image.file_format = output_format
            cube_rgb_image.colorspace_settings.name = 'sRGB' if not is_linear else 'Non-Color'

            # Combine RGB channels with alpha channel set to 1
            cube_rgb_alpha = np.dstack((cube_rgb, np.ones_like(cube_alpha)))

            # Flatten and assign pixels
            cube_rgb_image.pixels = cube_rgb_alpha.flatten().tolist()

            # Save the RGB cubemap image
            dir_name = os.path.dirname(equirectangular_image_path)
            base_name = os.path.basename(equirectangular_image_path)
            file_name, _ = os.path.splitext(base_name)
            cube_rgb_file_name = f"{file_name}_EquiToCube_rgb{ext}"
            cube_rgb_image_path = os.path.join(dir_name, cube_rgb_file_name)
            cube_rgb_image.filepath_raw = cube_rgb_image_path
            cube_rgb_image.save()

            print(f"Saved RGB cubemap image to: {cube_rgb_image_path}")

            # Create Alpha cubemap image with alpha channel set to 1
            cube_alpha_image = create_or_replace_image(
                "Cubemap Alpha Image",
                width=width_c,
                height=height_c,
                alpha=True,
                float_buffer=float_buffer
            )
            cube_alpha_image.use_half_precision = use_half_precision
            cube_alpha_image.file_format = output_format
            cube_alpha_image.colorspace_settings.name = 'Non-Color'  # Alpha is linear

            # Replace RGB channels with alpha data, set alpha channel to 1
            cube_alpha_rgb = np.dstack((cube_alpha, cube_alpha, cube_alpha, np.ones_like(cube_alpha)))

            # Flatten and assign pixels
            cube_alpha_image.pixels = cube_alpha_rgb.flatten().tolist()

            # Save the Alpha cubemap image
            cube_alpha_file_name = f"{file_name}_EquiToCube_alpha{ext}"
            cube_alpha_image_path = os.path.join(dir_name, cube_alpha_file_name)
            cube_alpha_image.filepath_raw = cube_alpha_image_path
            cube_alpha_image.save()

            print(f"Saved Alpha cubemap image to: {cube_alpha_image_path}")

        else:
            # Combine RGB and alpha channels
            cube_rgba = np.dstack((cube_rgb, cube_alpha))

            # Create combined cubemap image
            cubemap_image = create_or_replace_image(
                "Cubemap Image",
                width=width_c,
                height=height_c,
                alpha=True,
                float_buffer=float_buffer
            )
            cubemap_image.use_half_precision = use_half_precision
            cubemap_image.file_format = output_format
            cubemap_image.colorspace_settings.name = 'sRGB' if not is_linear else 'Non-Color'

            # Flatten and assign pixels
            cubemap_image.pixels = cube_rgba.flatten().tolist()

            # Save the image
            dir_name = os.path.dirname(equirectangular_image_path)
            base_name = os.path.basename(equirectangular_image_path)
            file_name, _ = os.path.splitext(base_name)
            cubemap_file_name = f"{file_name}_EquiToCube{ext}"
            cubemap_image_path = os.path.join(dir_name, cubemap_file_name)
            cubemap_image.filepath_raw = cubemap_image_path
            cubemap_image.save()

            print(f"Saved cubemap image to: {cubemap_image_path}")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        import traceback
        traceback.print_exc()


def generate_dice_mask(height, width):
    """
    Generates a boolean mask for a standard 4x3 dice layout.
    True = Valid Face, False = Background/Empty
    """
    face_size = height // 3
    mask = np.zeros((height, width), dtype=bool)

    # Dice Layout (Horizontal Cross)
    # Row 0: Empty, UP, Empty, Empty
    # Row 1: LEFT, FRONT, RIGHT, BACK
    # Row 2: Empty, DOWN, Empty, Empty

    # Indices (col, row) for faces
    # faces = [
    #     ('U', 1, 0),
    #     ('L', 0, 1), ('F', 1, 1), ('R', 2, 1), ('B', 3, 1),
    #     ('D', 1, 2)
    # ]
    
    faces_indices = [
        (1, 0), # Up
        (0, 1), (1, 1), (2, 1), (3, 1), # Left, Front, Right, Back
        (1, 2)  # Down
    ]

    for col, row in faces_indices:
        mask[row*face_size : (row+1)*face_size, col*face_size : (col+1)*face_size] = True

    return mask

def inpaint_image(img, mask, erosion=0):
    """
    Fills invalid (False) regions in the image using Nearest Neighbor inpainting.
    Uses scipy.ndimage.distance_transform_edt to find indices of nearest valid pixels.
    Optional erosion argument shrinks the mask to overwrite edge pixels.
    If erosion is 0, inpainting is skipped.
    """
    if erosion == 0:
        return img

    if erosion > 0:
        print(f"Eroding mask by {erosion} pixels...")
        mask = scipy.ndimage.binary_erosion(mask, iterations=erosion)

    if np.all(mask):
        return img

    print("Inpainting background regions...")
    
    # We need indices of the nearest foreground pixel for every background pixel.
    # distance_transform_edt computes distance to nearest zero-value pixel.
    # So we pass the INVERTED mask (Background=1, Foreground=0).
    # return_indices=True gives us the indices of that nearest zero-pixel (Foreground).
    
    # We compute this for the first channel (mask is 2D), then apply to all channels.
    distances, indices = scipy.ndimage.distance_transform_edt(~mask, return_distances=True, return_indices=True)
    
    # indices is (2, H, W) -> [row_indices, col_indices]
    row_indices = indices[0]
    col_indices = indices[1]
    
    inpainted_img = img.copy()
    
    # Fill invalid pixels using the gathered indices (NN Initialization)
    if img.ndim == 2:
        inpainted_img = img[row_indices, col_indices]
    else:
        # We apply the same spatial mapping to each channel
        for c in range(img.shape[2]):
             channel = img[:, :, c]
             inpainted_img[:, :, c] = channel[row_indices, col_indices]
    
    # --------------------------------------------------------
    # Smooth Diffusion Step (Simulating Telea/Navier-Stokes)
    # --------------------------------------------------------
    # We use the NN result as a solid initialization, then iteratively blur the
    # invalid regions to smooth out the "streaky" artifacts of NN.
    # This creates a result closer to fluid-based inpainting.
    
    diffusion_steps = 15  # Number of smoothing iterations
    sigma = 1.5           # Blur radius
    
    # Only diffuse if we actually have invalid regions (which we do if we are here)
    if diffusion_steps > 0:
        print(f"Applying smooth diffusion ({diffusion_steps} steps)...")
        
        # Working buffer
        smoothed = inpainted_img.copy()
        
        for i in range(diffusion_steps):
            # Blur the entire image (Scipy's gaussian_filter is efficient)
            if img.ndim == 2:
                blurred = scipy.ndimage.gaussian_filter(smoothed, sigma=sigma)
            else:
                 # Blur per channel? multidimensional gaussian_filter handles 3D if we specify axis?
                 # Default blurs all axes. We want to blur Y and X, but NOT Channels (axis 2).
                 blurred = scipy.ndimage.gaussian_filter(smoothed, sigma=[sigma, sigma, 0])

            # The Core Logic:
            # 1. In the invalid region (where we want inpainting), take the BLURRED value.
            # 2. In the valid region (original image), KEEP the original value STRICTLY.
            #    (Actually, we keep the previous iteration's valid pixels... which are constant)
            
            # Update invalid regions with the blurred version of the previous step.
            # This propagates smooth colors from the valid boundary outwards.
            smoothed[~mask] = blurred[~mask]
            
            # Ensure valid regions remain untouched (reset to original/NN valid pixels)
            # Actually, `inpainted_img` has the original valid pixels in the `mask` area anyway.
            # So we only need to write to ~mask.
            # But the blur bleeds valid into invalid, and invalid (streaks) into invalid.
            # Over time, streaks dissolve into gradients.
        
        return smoothed
        
    return inpainted_img


def process_cubemap_to_equirectangular_data(cubemap_pixels, width, height, channels, is_linear, separate_alpha_channel, erosion=0):
    """
    Pure data processing for Cubemap -> Equirectangular conversion.
    Returns: (equirect_rgb, equirect_alpha, equirect_width, equirect_height) numpy arrays/values.
    """
    try:
        # Inpaint background to prevent edge seams
        try:
            dice_mask = generate_dice_mask(height, width)
            cubemap_pixels = inpaint_image(cubemap_pixels, dice_mask, erosion=erosion)
        except Exception as e:
            print(f"Warning: Inpainting failed: {e}. Proceeding without inpainting.")

        # Convert sRGB to linear if necessary
        if not is_linear:
            cubemap_pixels[:, :, :3] = srgb_to_linear(cubemap_pixels[:, :, :3])

        # Separate alpha channel if needed
        if channels == 4:
            alpha_cubemap = cubemap_pixels[:, :, 3]
            rgb_cubemap = cubemap_pixels[:, :, :3]
        else:
            alpha_cubemap = np.ones((height, width), dtype=np.float32)
            rgb_cubemap = cubemap_pixels[:, :, :3]

        # Determine output dimensions
        equirect_width = width // 4 * 8  # Equirectangular width is typically 2:1 ratio
        equirect_height = height // 3 * 4

        # Convert RGB cubemap to equirectangular
        equirect_rgb = py360convert.c2e(rgb_cubemap, h=equirect_height, w=equirect_width, cube_format='dice')

        # Convert alpha cubemap to equirectangular
        alpha_cubemap_expanded = np.stack([alpha_cubemap]*3, axis=-1)
        equirect_alpha = py360convert.c2e(alpha_cubemap_expanded, h=equirect_height, w=equirect_width, cube_format='dice')[:, :, 0]

        # Convert linear to sRGB if saving in sRGB format
        if not is_linear:
            equirect_rgb = linear_to_srgb(equirect_rgb)

        return equirect_rgb, equirect_alpha, equirect_width, equirect_height

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

def convert_cubemap_to_equirectangular(cubemap_image_path, separate_alpha_channel, erosion=0):
    print(f"Processing cubemap image: {cubemap_image_path}")
    try:
        # Load the cubemap image
        try:
            cubemap_image = bpy.data.images.load(cubemap_image_path)
            # Ensure it's packed or loaded? load() usually suffices.
        except Exception as e:
            msg = f"Failed to load image {cubemap_image_path}: {e}"
            print(msg)
            return False, msg


        print("Image loaded successfully.")

        # Determine the image color space and format based on file extension
        ext = os.path.splitext(cubemap_image_path)[1].lower()
        if ext in ['.exr', '.hdr']:
            is_linear = True
            output_format = 'OPEN_EXR'
            cubemap_image.colorspace_settings.name = 'Non-Color'
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            is_linear = False
            output_format = ext.replace('.', '').upper()
            if output_format == 'JPG':
                output_format = 'JPEG'
            elif output_format == 'TIF':
                output_format = 'TIFF'
            cubemap_image.colorspace_settings.name = 'Non-Color'  # Load without color management
        else:
            msg = f"Unsupported input image format: {ext}"
            print(msg)
            return False, msg

        width, height = cubemap_image.size
        channels = len(cubemap_image.pixels) // (width * height)

        print(f"Image size: width={width}, height={height}, channels={channels}")

        # Validation: Check if dimensions match standard 4x3 dice layout
        # cube_dice2h assumes height / 3 is the face size, and width should be 4 * face size.
        face_size = height // 3
        expected_width = face_size * 4
        
        # Allow a small tolerance for imperfect inputs or offer a better error
        if width != expected_width:
             message = f"Invalid cubemap dimensions. The addon expects a horizontal cross 'dice' layout (4x3 aspect ratio). Your image is {width}x{height}, but expected {expected_width}x{height} (based on height)."
             print(f"Error: {message}")
             raise ValueError(message)

        # Read the pixels and reshape
        # Must make a copy for threads if we were threading here, but this is the sync function.
        cubemap_pixels = np.array(cubemap_image.pixels[:]).reshape((height, width, channels)).astype(np.float32)
        cubemap_pixels = cubemap_pixels[:, :, :4]  # Ensure RGBA

        # --- Delegate Math to Pure Function ---
        equirect_rgb, equirect_alpha, equirect_width, equirect_height = process_cubemap_to_equirectangular_data(
            cubemap_pixels, width, height, channels, is_linear, separate_alpha_channel, erosion
        )
        # --------------------------------------

        # Clamp values between 0 and 1 for 8-bit formats
        if output_format in ['PNG', 'JPEG', 'TIFF', 'BMP']:
            equirect_rgb = np.clip(equirect_rgb, 0.0, 1.0)
            
        # ... Saving logic remains mostly same ...
        
        # Determine float_buffer and use_half_precision settings
        if output_format == 'OPEN_EXR':
            float_buffer = True
            use_half_precision = True
        else:
            float_buffer = False
            use_half_precision = False

        if separate_alpha_channel:
            # Create RGB equirectangular image with alpha channel set to 1
            equirect_rgb_image = create_or_replace_image(
                "Equirectangular RGB Image",
                width=equirect_width,
                height=equirect_height,
                alpha=True,
                float_buffer=float_buffer
            )
            equirect_rgb_image.use_half_precision = use_half_precision
            equirect_rgb_image.file_format = output_format
            equirect_rgb_image.colorspace_settings.name = 'sRGB' if not is_linear else 'Non-Color'
            
            # Combine RGB channels with alpha channel set to 1
            equirect_rgb_alpha = np.dstack((equirect_rgb, np.ones_like(equirect_alpha)))
            
            # Flatten and assign pixels
            equirect_rgb_image.pixels = equirect_rgb_alpha.flatten().tolist()
            
            # Save the RGB equirectangular image
            dir_name = os.path.dirname(cubemap_image_path)
            base_name = os.path.basename(cubemap_image_path)
            file_name, _ = os.path.splitext(base_name)
            equirect_rgb_file_name = f"{file_name}_CubeToEqui_rgb{ext}"
            equirect_rgb_image_path = os.path.join(dir_name, equirect_rgb_file_name)
            equirect_rgb_image.filepath_raw = equirect_rgb_image_path
            equirect_rgb_image.save()
            
            print(f"Saved RGB equirectangular image to: {equirect_rgb_image_path}")
            
            # Create Alpha equirectangular image with alpha channel set to 1
            equirect_alpha_image = create_or_replace_image(
                "Equirectangular Alpha Image",
                width=equirect_width,
                height=equirect_height,
                alpha=True,
                float_buffer=float_buffer
            )
            equirect_alpha_image.use_half_precision = use_half_precision
            equirect_alpha_image.file_format = output_format
            equirect_alpha_image.colorspace_settings.name = 'Non-Color'  # Alpha is linear
            
            # Replace RGB channels with alpha data, set alpha channel to 1
            equirect_alpha_rgb = np.dstack((equirect_alpha, equirect_alpha, equirect_alpha, np.ones_like(equirect_alpha)))
            
            # Flatten and assign pixels
            equirect_alpha_image.pixels = equirect_alpha_rgb.flatten().tolist()
            
            # Save the Alpha equirectangular image
            equirect_alpha_file_name = f"{file_name}_CubeToEqui_alpha{ext}"
            equirect_alpha_image_path = os.path.join(dir_name, equirect_alpha_file_name)
            equirect_alpha_image.filepath_raw = equirect_alpha_image_path
            equirect_alpha_image.save()
            
            print(f"Saved Alpha equirectangular image to: {equirect_alpha_image_path}")

        else:
            # Combine RGB and alpha channels
            equirect_rgba = np.dstack((equirect_rgb, equirect_alpha))
            
            # Create combined equirectangular image
            equirect_image = create_or_replace_image(
                "Equirectangular Image",
                width=equirect_width,
                height=equirect_height,
                alpha=True,
                float_buffer=float_buffer
            )
            equirect_image.use_half_precision = use_half_precision
            equirect_image.file_format = output_format
            equirect_image.colorspace_settings.name = 'sRGB' if not is_linear else 'Non-Color'
            
            # Flatten and assign pixels
            equirect_image.pixels = equirect_rgba.flatten().tolist()
            
            # Save the image
            dir_name = os.path.dirname(cubemap_image_path)
            base_name = os.path.basename(cubemap_image_path)
            file_name, _ = os.path.splitext(base_name)
            equirect_file_name = f"{file_name}_CubeToEqui{ext}"
            equirect_image_path = os.path.join(dir_name, equirect_file_name)
            equirect_image.filepath_raw = equirect_image_path
            equirect_image.save()

        print(f"Saved equirectangular image to: {equirect_image_path}")

        return True, f"Successfully converted {cubemap_image_path}"

    except Exception as e:
        msg = f"An error occurred during conversion: {e}"
        print(msg)
        import traceback
        traceback.print_exc()
        return False, str(e)

# We will make the Single operators inherit from the Batch ones or just duplicate logic for simplicity to avoid complex inheritance now. 
# Duplication is safer to avoid breaking changes in logic flow.

def create_or_replace_image(name, width, height, alpha=True, float_buffer=False):
    """
    Creates a new image, removing any existing image with the same name first.
    """
    if name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[name])
    
    img = bpy.data.images.new(
        name,
        width=width,
        height=height,
        alpha=alpha,
        float_buffer=float_buffer
    )
    return img

def set_image_in_editor(image):
    """
    Sets the given image in the first found Image Editor space.
    """
    for area in bpy.context.screen.areas:
        if area.type == 'IMAGE_EDITOR':
            for space in area.spaces:
                if space.type == 'IMAGE_EDITOR':
                    space.image = image
                    return # Set in one and done

class ConvertCubemapToEquirectangularOperator(bpy.types.Operator):
    bl_idname = "addon.convert_cubemap"
    bl_label = "Convert Cubemap to Equirectangular"
    
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.exr;*.hdr;*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp", options={'HIDDEN'})

    _timer = None
    _executor = None
    _future = None
    _state = 'IDLE'
    _current_file_data = None
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
        
    def execute(self, context):
        cubemap_image_path = self.filepath
        self.separate_alpha_channel = context.scene.separate_alpha_channel
        self.erosion = context.scene.inpainting_erosion
        
        if not cubemap_image_path or not os.path.exists(cubemap_image_path):
             self.report({'ERROR'}, "Invalid File Path")
             return {'CANCELLED'}

        self._state = 'IDLE'
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.progress_begin(0, 1) # Just 0-1
        context.scene.is_converting = True
        
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            return self.cancel(context)

        if event.type == 'TIMER':
            if self._state == 'IDLE':
                # Just one file to process
                file_path = self.filepath # Use self.filepath set in execute
                self._current_file_data = {'path': file_path}
                
                try:
                    img = bpy.data.images.load(file_path)
                    width, height = img.size
                    channels = len(img.pixels) // (width * height)
                    
                    face_size = height // 3
                    expected_width = face_size * 4
                    if width != expected_width:
                        raise ValueError(f"Invalid dimensions {width}x{height}")
                    
                    ext = os.path.splitext(file_path)[1].lower()
                    is_linear = ext in ['.exr', '.hdr']
                    if is_linear:
                        img.colorspace_settings.name = 'Non-Color'
                        output_format = 'OPEN_EXR'
                    else:
                        output_format = 'PNG'
                        if ext in ['.jpg', '.jpeg']: output_format = 'JPEG'
                        elif ext == '.tiff': output_format = 'TIFF'
                        elif ext == '.bmp': output_format = 'BMP'
                        img.colorspace_settings.name = 'Non-Color'
                    
                    self._current_file_data['is_linear'] = is_linear
                    self._current_file_data['output_format'] = output_format
                    self._current_file_data['ext'] = ext

                    pixels = np.array(img.pixels[:]).reshape((height, width, channels)).astype(np.float32)
                    pixels = pixels[:, :, :4] # RGBA

                    self._future = self._executor.submit(
                        process_cubemap_to_equirectangular_data,
                        pixels, width, height, channels, 
                        is_linear, self.separate_alpha_channel, self.erosion
                    )
                    self._state = 'COMPUTING'
                    bpy.data.images.remove(img)

                except Exception as e:
                    self.report({'ERROR'}, f"Failed to load: {e}")
                    return self.cancel(context)
                
                # If we passed here, we are computing.
                # If we failed (and didn't return), we should stop?
                # The try/catch returns cancel on error, so we are good.

            elif self._state == 'COMPUTING':
                if self._future.done():
                    try:
                        equirect_rgb, equirect_alpha, w, h = self._future.result()
                        
                        # Save Result code is duplicated from the batch operator, 
                        # ideally we'd make a mixin, but inline for now.
                        saved_name = self.save_result(
                            self._current_file_data, 
                            equirect_rgb, equirect_alpha, w, h,
                            self.separate_alpha_channel
                        )
                        self.report({'INFO'}, f"Converted: {saved_name}")
                        
                    except Exception as e:
                        self.report({'ERROR'}, f"Conversion failed: {e}")
                        import traceback
                        traceback.print_exc()

                    return self.finish(context)

        return {'PASS_THROUGH'}

    def save_result(self, file_data, equirect_rgb, equirect_alpha, width, height, separate_alpha):
        output_format = file_data['output_format']
        is_linear = file_data['is_linear']
        ext = file_data['ext']
        path = file_data['path']
        float_buffer = (output_format == 'OPEN_EXR')
        use_half = float_buffer # Simplify
        
        dir_name = os.path.dirname(path)
        base_name = os.path.basename(path)
        f_name, _ = os.path.splitext(base_name)

        if separate_alpha:
            internal_name_rgb = f"{f_name}_CubeToEqui_RGB"
            img_rgb = create_or_replace_image(internal_name_rgb, width, height, alpha=True, float_buffer=float_buffer)
            img_rgb.file_format = output_format
            img_rgb.use_half_precision = use_half
            img_rgb.colorspace_settings.name = 'sRGB' if not is_linear else 'Non-Color'
            pixels_rgb = np.dstack((equirect_rgb, np.ones_like(equirect_alpha))).flatten().tolist()
            img_rgb.pixels = pixels_rgb
            save_path = os.path.join(dir_name, f"{f_name}_CubeToEqui_rgb{ext}")
            img_rgb.filepath_raw = save_path
            img_rgb.save()
            
            set_image_in_editor(img_rgb)
            # bpy.data.images.remove(img_rgb) # Keep it open
            
            internal_name_alpha = f"{f_name}_CubeToEqui_Alpha"
            img_alpha = create_or_replace_image(internal_name_alpha, width, height, alpha=True, float_buffer=float_buffer)
            img_alpha.file_format = output_format
            img_alpha.use_half_precision = use_half
            img_alpha.colorspace_settings.name = 'Non-Color'
            pixels_alpha = np.dstack((equirect_alpha, equirect_alpha, equirect_alpha, np.ones_like(equirect_alpha))).flatten().tolist()
            img_alpha.pixels = pixels_alpha
            save_path_alpha = os.path.join(dir_name, f"{f_name}_CubeToEqui_alpha{ext}")
            img_alpha.filepath_raw = save_path_alpha
            img_alpha.save()
            # bpy.data.images.remove(img_alpha) # Keep option? Usually alpha is second, maybe just keep RGB visible.
            
            return os.path.basename(save_path)

        else:
            internal_name = f"{f_name}_CubeToEqui"
            img = create_or_replace_image(internal_name, width, height, alpha=True, float_buffer=float_buffer)
            img.file_format = output_format
            img.use_half_precision = use_half
            img.colorspace_settings.name = 'sRGB' if not is_linear else 'Non-Color'
            pixels = np.dstack((equirect_rgb, equirect_alpha)).flatten().tolist()
            img.pixels = pixels
            save_path = os.path.join(dir_name, f"{f_name}_CubeToEqui{ext}")
            img.filepath_raw = save_path
            img.save()
            
            set_image_in_editor(img)
            # bpy.data.images.remove(img) # Keep it open
            
            return os.path.basename(save_path)

    def cancel(self, context):
        if self._executor:
            self._executor.shutdown(wait=False)
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.window_manager.progress_end()
        context.scene.is_converting = False
        self.report({'WARNING'}, "Conversion cancelled")
        return {'CANCELLED'}

    def finish(self, context):
        if self._executor:
            self._executor.shutdown(wait=True)
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.window_manager.progress_end()
        context.scene.is_converting = False
        return {'FINISHED'}

class ConvertAllCubemapsToEquirectangularOperator(bpy.types.Operator):
    bl_idname = "addon.convert_all_cubemaps"
    bl_label = "Convert All Cubemaps in Directory"
    
    directory: bpy.props.StringProperty(subtype="DIR_PATH")
    filter_folder: bpy.props.BoolProperty(default=True, options={'HIDDEN'})
    
    _timer = None
    _files_to_process = []
    _total_files = 0
    _current_file_index = 0
    _count = 0
    _success_count = 0
    _fail_count = 0
    
    _executor = None
    _future = None
    _state = 'IDLE' 
    _current_file_data = None 

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
        
    def execute(self, context):
        directory = self.directory
        self.separate_alpha_channel = context.scene.separate_alpha_channel
        self.erosion = context.scene.inpainting_erosion

        if not directory or not os.path.exists(directory):
            self.report({'ERROR'}, "Invalid Directory")
            return {'CANCELLED'}

        self._files_to_process = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".hdr", ".exr", ".tif", ".tiff", ".bmp")):
                    self._files_to_process.append(os.path.join(root, file))

        if not self._files_to_process:
            self.report({'WARNING'}, "No compatible files found.")
            return {'CANCELLED'}

        self._total_files = len(self._files_to_process)
        self._current_file_index = 0
        self._count = 0
        self._success_count = 0
        self._fail_count = 0
        self._state = 'IDLE'
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.progress_begin(0, self._total_files)
        context.scene.is_converting = True 
        
        return {'RUNNING_MODAL'}
    
    def modal(self, context, event):
        if event.type == 'ESC':
            return self.cancel(context)

        if event.type == 'TIMER':
            if self._state == 'IDLE':
                if self._files_to_process:
                    file_path = self._files_to_process.pop(0)
                    self._current_file_data = {'path': file_path}
                    
                    # Update progress
                    current_index = self._total_files - len(self._files_to_process)
                    context.window_manager.progress_update(current_index)
                    
                    try:
                        img = bpy.data.images.load(file_path)
                        width, height = img.size
                        channels = len(img.pixels) // (width * height)
                        
                        # Validation for Cubemap
                        face_size = height // 3
                        expected_width = face_size * 4
                        if width != expected_width:
                             print(f"Skipping {file_path}: Invalid expected dimensions.")
                             self._state = 'IDLE' 
                             return {'PASS_THROUGH'}

                        ext = os.path.splitext(file_path)[1].lower()
                        is_linear = ext in ['.exr', '.hdr']
                        if is_linear:
                            img.colorspace_settings.name = 'Non-Color'
                            output_format = 'OPEN_EXR'
                        else:
                            output_format = 'PNG'
                            if ext in ['.jpg', '.jpeg']: output_format = 'JPEG'
                            elif ext == '.tiff': output_format = 'TIFF'
                            elif ext == '.bmp': output_format = 'BMP'
                            img.colorspace_settings.name = 'Non-Color'
                            
                        self._current_file_data['is_linear'] = is_linear
                        self._current_file_data['output_format'] = output_format
                        self._current_file_data['ext'] = ext

                        pixels = np.array(img.pixels[:]).reshape((height, width, channels)).astype(np.float32)
                        pixels = pixels[:, :, :4] # RGBA
                        
                        self._future = self._executor.submit(
                            process_cubemap_to_equirectangular_data,
                            pixels, width, height, channels, 
                            is_linear, self.separate_alpha_channel, self.erosion
                        )
                        self._state = 'COMPUTING'
                        bpy.data.images.remove(img)

                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        self._state = 'IDLE'

                else:
                    return self.finish(context)

            elif self._state == 'COMPUTING':
                if self._future.done():
                    try:
                        equirect_rgb, equirect_alpha, w, h = self._future.result()
                        
                        self.save_result(
                            self._current_file_data, 
                            equirect_rgb, equirect_alpha, w, h,
                            self.separate_alpha_channel
                        )
                        self._success_count += 1
                    except Exception as e:
                        print(f"Error processing {self._current_file_data.get('path')}: {e}")
                        self._fail_count += 1
                    
                    self._state = 'IDLE'
                    self._future = None
                    self._current_file_data = None

        return {'PASS_THROUGH'}

    def save_result(self, file_data, equirect_rgb, equirect_alpha, width, height, separate_alpha):
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
            img_rgb = create_or_replace_image(f"Temp_RGB", width, height, alpha=True, float_buffer=float_buffer)
            img_rgb.file_format = output_format
            img_rgb.use_half_precision = use_half
            img_rgb.colorspace_settings.name = 'sRGB' if not is_linear else 'Non-Color'
            pixels_rgb = np.dstack((equirect_rgb, np.ones_like(equirect_alpha))).flatten().tolist()
            img_rgb.pixels = pixels_rgb
            save_path = os.path.join(dir_name, f"{f_name}_CubeToEqui_rgb{ext}")
            img_rgb.filepath_raw = save_path
            img_rgb.save()
            bpy.data.images.remove(img_rgb)
            
            img_alpha = create_or_replace_image(f"Temp_Alpha", width, height, alpha=True, float_buffer=float_buffer)
            img_alpha.file_format = output_format
            img_alpha.use_half_precision = use_half
            img_alpha.colorspace_settings.name = 'Non-Color'
            pixels_alpha = np.dstack((equirect_alpha, equirect_alpha, equirect_alpha, np.ones_like(equirect_alpha))).flatten().tolist()
            img_alpha.pixels = pixels_alpha
            save_path = os.path.join(dir_name, f"{f_name}_CubeToEqui_alpha{ext}")
            img_alpha.filepath_raw = save_path
            img_alpha.save()
            bpy.data.images.remove(img_alpha)

        else:
            img = create_or_replace_image(f"Temp_Comb", width, height, alpha=True, float_buffer=float_buffer)
            img.file_format = output_format
            img.use_half_precision = use_half
            img.colorspace_settings.name = 'sRGB' if not is_linear else 'Non-Color'
            pixels = np.dstack((equirect_rgb, equirect_alpha)).flatten().tolist()
            img.pixels = pixels
            save_path = os.path.join(dir_name, f"{f_name}_CubeToEqui{ext}")
            img.filepath_raw = save_path
            img.save()
            bpy.data.images.remove(img)

    def cancel(self, context):
        if self._executor:
            self._executor.shutdown(wait=False)
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.window_manager.progress_end()
        context.scene.is_converting = False
        self.report({'WARNING'}, "Batch conversion cancelled.")
        return {'CANCELLED'}

    def finish(self, context):
        if self._executor:
            self._executor.shutdown(wait=True)
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.window_manager.progress_end()
        context.scene.is_converting = False
        
        if self._fail_count > 0:
            self.report({'WARNING'}, f"Processed {self._success_count} files. Failed: {self._fail_count}. Check console.")
        else:
            self.report({'INFO'}, f"Successfully converted all {self._success_count} cubemaps.")
            
        return {'FINISHED'}


class ConvertEquirectangularToCubemapOperator(bpy.types.Operator):
    bl_idname = "addon.convert_equirectangular"
    bl_label = "Convert Equirectangular to Cubemap"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.exr;*.hdr;*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp", options={'HIDDEN'})

    _timer = None
    _executor = None
    _future = None
    _state = 'IDLE'
    _current_file_data = None 
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        file_path = self.filepath
        self.separate_alpha_channel = context.scene.separate_alpha_channel
        self.erosion = context.scene.inpainting_erosion
        
        if not file_path or not os.path.exists(file_path):
             self.report({'ERROR'}, "Invalid File Path")
             return {'CANCELLED'}

        self._state = 'IDLE'
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.progress_begin(0, 1)
        context.scene.is_converting = True
        
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            return self.cancel(context)

        if event.type == 'TIMER':
            if self._state == 'IDLE':
                file_path = self.filepath # Use self.filepath
                self._current_file_data = {'path': file_path}
                
                try:
                    img = bpy.data.images.load(file_path)
                    width, height = img.size
                    channels = len(img.pixels) // (width * height)
                    
                    ext = os.path.splitext(file_path)[1].lower()
                    is_linear = ext in ['.exr', '.hdr']
                    if is_linear:
                        img.colorspace_settings.name = 'Non-Color'
                        output_format = 'OPEN_EXR'
                    else:
                        output_format = 'PNG'
                        if ext in ['.jpg', '.jpeg']: output_format = 'JPEG'
                        elif ext == '.tiff': output_format = 'TIFF'
                        elif ext == '.bmp': output_format = 'BMP'
                        img.colorspace_settings.name = 'Non-Color'
                        
                    self._current_file_data['is_linear'] = is_linear
                    self._current_file_data['output_format'] = output_format
                    self._current_file_data['ext'] = ext

                    pixels = np.array(img.pixels[:]).reshape((height, width, channels)).astype(np.float32)
                    pixels = pixels[:, :, :4] # RGBA
                    
                    self._future = self._executor.submit(
                        process_equirectangular_to_cubemap_data,
                        pixels, width, height, channels, is_linear, self.separate_alpha_channel, self.erosion
                    )
                    self._state = 'COMPUTING'
                    bpy.data.images.remove(img)

                except Exception as e:
                    self.report({'ERROR'}, f"Error loading: {e}")
                    return self.cancel(context)

            elif self._state == 'COMPUTING':
                if self._future.done():
                    try:
                        cube_rgb, cube_alpha, w, h = self._future.result()
                        
                        saved_name = self.save_result(
                            self._current_file_data, 
                            cube_rgb, cube_alpha, w, h,
                            self.separate_alpha_channel
                        )
                        self.report({'INFO'}, f"Converted: {saved_name}")
                    except Exception as e:
                        self.report({'ERROR'}, f"Error processing: {e}")
                    
                    return self.finish(context)

        return {'PASS_THROUGH'}

    def save_result(self, file_data, cube_rgb, cube_alpha, width, height, separate_alpha):
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
            img_rgb = bpy.data.images.new(f"Temp_RGB", width, height, alpha=True, float_buffer=float_buffer)
            img_rgb.file_format = output_format
            img_rgb.use_half_precision = use_half
            img_rgb.colorspace_settings.name = 'sRGB' if not is_linear else 'Non-Color'
            pixels_rgb = np.dstack((cube_rgb, np.ones_like(cube_alpha))).flatten().tolist()
            img_rgb.pixels = pixels_rgb
            save_path = os.path.join(dir_name, f"{f_name}_EquiToCube_rgb{ext}")
            img_rgb.filepath_raw = save_path
            img_rgb.save()
            bpy.data.images.remove(img_rgb)
            
            img_alpha = bpy.data.images.new(f"Temp_Alpha", width, height, alpha=True, float_buffer=float_buffer)
            img_alpha.file_format = output_format
            img_alpha.use_half_precision = use_half
            img_alpha.colorspace_settings.name = 'Non-Color'
            pixels_alpha = np.dstack((cube_alpha, cube_alpha, cube_alpha, np.ones_like(cube_alpha))).flatten().tolist()
            img_alpha.pixels = pixels_alpha
            save_path = os.path.join(dir_name, f"{f_name}_EquiToCube_alpha{ext}")
            img_alpha.filepath_raw = save_path
            img_alpha.save()
            bpy.data.images.remove(img_alpha)
        
        else:
            img = bpy.data.images.new(f"Temp_Comb", width, height, alpha=True, float_buffer=float_buffer)
            img.file_format = output_format
            img.use_half_precision = use_half
            img.colorspace_settings.name = 'sRGB' if not is_linear else 'Non-Color'
            pixels = np.dstack((cube_rgb, cube_alpha)).flatten().tolist()
            img.pixels = pixels
            save_path = os.path.join(dir_name, f"{f_name}_EquiToCube{ext}")
            img.filepath_raw = save_path
            img.save()
            bpy.data.images.remove(img)

    def cancel(self, context):
        if self._executor:
            self._executor.shutdown(wait=False)
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.window_manager.progress_end()
        context.scene.is_converting = False
        self.report({'WARNING'}, "Cancelled")
        return {'CANCELLED'}

    def finish(self, context):
        if self._executor:
            self._executor.shutdown(wait=True)
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.window_manager.progress_end()
        context.scene.is_converting = False
        return {'FINISHED'}


class ConvertAllEquirectangularsToCubemapOperator(bpy.types.Operator):
    bl_idname = "addon.convert_all_equirectangulars"
    bl_label = "Convert All Equirectangulars to Cubemap"

    directory: bpy.props.StringProperty(subtype="DIR_PATH")
    filter_folder: bpy.props.BoolProperty(default=True, options={'HIDDEN'})

    _timer = None
    _files_to_process = []
    _total_files = 0
    _count = 0 
    
    _executor = None
    _future = None
    _state = 'IDLE' 
    _current_file_data = None 

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        directory = self.directory
        self.separate_alpha_channel = context.scene.separate_alpha_channel
        self.erosion = context.scene.inpainting_erosion

        if not directory or not os.path.exists(directory):
            self.report({'ERROR'}, "Invalid Directory")
            return {'CANCELLED'}

        self._files_to_process = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".hdr", ".exr", ".tif", ".tiff", ".bmp")):
                    self._files_to_process.append(os.path.join(root, file))

        if not self._files_to_process:
            self.report({'WARNING'}, "No compatible files found.")
            return {'CANCELLED'}

        self._total_files = len(self._files_to_process)
        self._count = 0
        self._state = 'IDLE'
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.progress_begin(0, self._total_files)
        context.scene.is_converting = True
        
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            return self.cancel(context)

        if event.type == 'TIMER':
            if self._state == 'IDLE':
                if self._files_to_process:
                    file_path = self._files_to_process.pop(0)
                    self._current_file_data = {'path': file_path}
                    
                    # Update progress
                    current_index = self._total_files - len(self._files_to_process)
                    context.window_manager.progress_update(current_index)
                    
                    try:
                        img = bpy.data.images.load(file_path)
                        width, height = img.size
                        channels = len(img.pixels) // (width * height)
                        
                        ext = os.path.splitext(file_path)[1].lower()
                        is_linear = ext in ['.exr', '.hdr']
                        if is_linear:
                            img.colorspace_settings.name = 'Non-Color'
                            output_format = 'OPEN_EXR'
                        else:
                            output_format = 'PNG'
                            if ext in ['.jpg', '.jpeg']: output_format = 'JPEG'
                            elif ext == '.tiff': output_format = 'TIFF'
                            elif ext == '.bmp': output_format = 'BMP'
                            img.colorspace_settings.name = 'Non-Color'
                            
                        self._current_file_data['is_linear'] = is_linear
                        self._current_file_data['output_format'] = output_format
                        self._current_file_data['ext'] = ext

                        pixels = np.array(img.pixels[:]).reshape((height, width, channels)).astype(np.float32)
                        pixels = pixels[:, :, :4] # RGBA
                        
                        self._future = self._executor.submit(
                            process_equirectangular_to_cubemap_data,
                            pixels, width, height, channels, is_linear, self.separate_alpha_channel, self.erosion
                        )
                        self._state = 'COMPUTING'
                        bpy.data.images.remove(img)

                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        self._state = 'IDLE'

                else:
                    return self.finish(context)

            elif self._state == 'COMPUTING':
                if self._future.done():
                    try:
                        cube_rgb, cube_alpha, w, h = self._future.result()
                        
                        self.save_result(
                            self._current_file_data, 
                            cube_rgb, cube_alpha, w, h,
                            self.separate_alpha_channel
                        )
                        self._count += 1
                    except Exception as e:
                        print(f"Error processing {self._current_file_data.get('path')}: {e}")
                    
                    self._state = 'IDLE'
                    self._future = None
                    self._current_file_data = None

        return {'PASS_THROUGH'}
    
    def save_result(self, file_data, cube_rgb, cube_alpha, width, height, separate_alpha):
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
             # RGB
            internal_name_rgb = f"{f_name}_EquiToCube_RGB"
            img_rgb = create_or_replace_image(internal_name_rgb, width, height, alpha=True, float_buffer=float_buffer)
            img_rgb.file_format = output_format
            img_rgb.use_half_precision = use_half
            img_rgb.colorspace_settings.name = 'sRGB' if not is_linear else 'Non-Color'
            
            pixels_rgb = np.dstack((cube_rgb, np.ones_like(cube_alpha))).flatten().tolist()
            img_rgb.pixels = pixels_rgb
            
            save_path = os.path.join(dir_name, f"{f_name}_EquiToCube_rgb{ext}")
            img_rgb.filepath_raw = save_path
            img_rgb.save()
            
            # set_image_in_editor(img_rgb)
            bpy.data.images.remove(img_rgb)
            
            # Alpha
            internal_name_alpha = f"{f_name}_EquiToCube_Alpha"
            img_alpha = create_or_replace_image(internal_name_alpha, width, height, alpha=True, float_buffer=float_buffer)
            img_alpha.file_format = output_format
            img_alpha.use_half_precision = use_half
            img_alpha.colorspace_settings.name = 'Non-Color'
            
            pixels_alpha = np.dstack((cube_alpha, cube_alpha, cube_alpha, np.ones_like(cube_alpha))).flatten().tolist()
            img_alpha.pixels = pixels_alpha
            
            save_path = os.path.join(dir_name, f"{f_name}_EquiToCube_alpha{ext}")
            img_alpha.filepath_raw = save_path
            img_alpha.save()
            # bpy.data.images.remove(img_alpha)
            
            return os.path.basename(save_path)
        
        else:
             # Combined
            internal_name = f"{f_name}_EquiToCube"
            img = create_or_replace_image(internal_name, width, height, alpha=True, float_buffer=float_buffer)
            img.file_format = output_format
            img.use_half_precision = use_half
            img.colorspace_settings.name = 'sRGB' if not is_linear else 'Non-Color'
            
            pixels = np.dstack((cube_rgb, cube_alpha)).flatten().tolist()
            img.pixels = pixels
            
            save_path = os.path.join(dir_name, f"{f_name}_EquiToCube{ext}")
            img.filepath_raw = save_path
            img.save()
            
            set_image_in_editor(img)
            # bpy.data.images.remove(img)
            
            return os.path.basename(save_path)


    def cancel(self, context):
        if self._executor:
            self._executor.shutdown(wait=False)
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.window_manager.progress_end()
        context.scene.is_converting = False
        self.report({'WARNING'}, "Operation cancelled")
        return {'CANCELLED'}

    def finish(self, context):
        if self._executor:
            self._executor.shutdown(wait=True)
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.window_manager.progress_end()
        context.scene.is_converting = False
        self.report({'INFO'}, f"Converted {self._count} equirectangular images.")
        return {'FINISHED'}

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