from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera


def prepare_scene(args, filename, cam=None):
    # Scene Setup
    scene = load_scene(filename)

    # in Hz; implicitly updates RadioMaterials
    scene.frequency = args.frequency
    # If set to False, ray tracing will be done per antenna element (slower for large arrays)
    scene.synthetic_array = args.synthetic_array

    if cam is not None:
        scene.add(cam)

    # Device Setup
    scene.tx_array = PlanarArray(
        num_rows=args.tx_num_rows,
        num_cols=args.tx_num_cols,
        vertical_spacing=args.tx_vertical_spacing,
        horizontal_spacing=args.tx_horizontal_spacing,
        pattern=args.tx_pattern,
        polarization=args.tx_polarization,
    )
    tx = Transmitter("tx", args.tx_position, args.tx_orientation)
    scene.add(tx)

    scene.rx_array = PlanarArray(
        num_rows=args.rx_num_rows,
        num_cols=args.rx_num_cols,
        vertical_spacing=args.rx_vertical_spacing,
        horizontal_spacing=args.rx_horizontal_spacing,
        pattern=args.rx_pattern,
        polarization=args.rx_polarization,
    )
    if args.rx_included:
        rx = Receiver("rx", args.rx_position, args.rx_orientation)
        scene.add(rx)

    return scene


def prepare_camera(args):
    cam = Camera(
        "my_cam",
        position=args.cam_position,
        orientation=args.cam_orientation,
    )
    cam.look_at(args.cam_look_at)
    return cam
