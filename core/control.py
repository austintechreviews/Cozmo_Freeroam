import pycozmo
import time
from PIL import Image
import numpy as np

def wakeup(cli):   
    print("Waking up the robot.")
    cli.conn.send(pycozmo.protocol_encoder.EnableStopOnCliff(enable=False))
    cli.set_lift_height(1.0)  # lift fully up to clear charger lip, not needed
    cli.drive_wheels(lwheel_speed=40, rwheel_speed=40, duration=2.75) # issues in regards to cliff sensor due to lip on charger
    cli.set_lift_height(0.0)
    cli.set_head_angle(0.0)
    cli.stop_all_motors()
    cli.conn.send(pycozmo.protocol_encoder.EnableStopOnCliff(enable=True))

def reverse_to_charger(cli, charger_rel_angle): #placeholder function, needs to be filled in
    print("reversing to charger")
    # turn to face away from charger

def show_expression(cli, expression, head_angle=None, hold_seconds=1):
    """
    Display a procedural face expression on Cozmo.

    - expression: instance of a pycozmo.expressions.* class, an int index, or a string
      matching the expression class name (case-insensitive).
    - head_angle: float radians for head angle. If None, use a reasonable default.
    - hold_seconds: how many seconds to hold the expression before returning to neutral.
    """
    # Build list of available expressions
    expressions = [
        pycozmo.expressions.Anger,
        pycozmo.expressions.Sadness,
        pycozmo.expressions.Happiness,
        pycozmo.expressions.Surprise,
        pycozmo.expressions.Disgust,
        pycozmo.expressions.Fear,
        pycozmo.expressions.Pleading,
        pycozmo.expressions.Vulnerability,
        pycozmo.expressions.Despair,
        pycozmo.expressions.Guilt,
        pycozmo.expressions.Disappointment,
        pycozmo.expressions.Embarrassment,
        pycozmo.expressions.Horror,
        pycozmo.expressions.Skepticism,
        pycozmo.expressions.Annoyance,
        pycozmo.expressions.Fury,
        pycozmo.expressions.Suspicion,
        pycozmo.expressions.Rejection,
        pycozmo.expressions.Boredom,
        pycozmo.expressions.Tiredness,
        pycozmo.expressions.Asleep,
        pycozmo.expressions.Confusion,
        pycozmo.expressions.Amazement,
        pycozmo.expressions.Excitement,
    ]

    # Resolve expression argument to an instance
    expr_instance = None
    if isinstance(expression, str):
        name = expression.strip().lower()
        for cls in expressions:
            if cls.__name__.lower() == name:
                expr_instance = cls()
                break
        if expr_instance is None:
            raise ValueError(f"Unknown expression name: {expression}")
    elif isinstance(expression, int):
        idx = expression
        if idx < 0 or idx >= len(expressions):
            raise IndexError("Expression index out of range")
        expr_instance = expressions[idx]()
    else:
        # Assume it's already an instance
        expr_instance = expression

    base_face = pycozmo.expressions.Neutral()

    # Default head angle (same logic as original example)
    if head_angle is None:
        head_angle = (pycozmo.robot.MAX_HEAD_ANGLE.radians - pycozmo.robot.MIN_HEAD_ANGLE.radians) / 2.0

    rate = pycozmo.robot.FRAME_RATE
    # Ensure interpolation frames at least 1
    interp_frames = max(1, rate // 3)

    
    # Set head angle
    cli.set_head_angle(head_angle)
    time.sleep(0.1)

    timer = pycozmo.util.FPSTimer(rate)

        # Transition to expression and back
    for from_face, to_face in ((base_face, expr_instance), (expr_instance, base_face)):

        if to_face != base_face:
            print(to_face.__class__.__name__)

        face_generator = pycozmo.procedural_face.interpolate(from_face, to_face, interp_frames)
        for face in face_generator:
            im = face.render()
            np_im = np.array(im)
            np_im2 = np_im[::2]  # Cozmo expects 128x32 (take even lines)
            im2 = Image.fromarray(np_im2)
            cli.display_image(im2)
            timer.sleep()

        # Pause for hold_seconds after reaching target expression (only once)
        if to_face == expr_instance:
            for _ in range(rate * max(0, hold_seconds)):
                timer.sleep()

def camera_stream(cli):


with pycozmo.connect(enable_procedural_face=False) as cli:
    wakeup(cli)
    show_expression(cli, "Happiness", head_angle=1.0, hold_seconds=2)