from __future__ import annotations

import click

from yolo_two_stage.pipeline.two_stage import TwoStagePipeline


@click.command()
@click.option("--stage1-weights", required=True, type=click.Path(exists=True, dir_okay=False), help="Stage-1 YOLO weights (vehicle vs living_beings)")
@click.option("--stage2-weights", required=True, type=click.Path(exists=True, dir_okay=False), help="Stage-2 YOLO weights (vehicle types)")
@click.option("--input", "input_image", required=True, type=click.Path(exists=True, dir_okay=False), help="Input image path")
@click.option("--output", required=True, type=click.Path(dir_okay=False), help="Output image path")
@click.option("--device", default=None, show_default=True, help="Device for inference, e.g. 'cpu', 'cuda:0'")
@click.option("--stage1-conf", default=0.25, show_default=True, type=float)
@click.option("--stage2-conf", default=0.25, show_default=True, type=float)
@click.option("--stage1-iou", default=0.45, show_default=True, type=float)
@click.option("--stage2-iou", default=0.45, show_default=True, type=float)
@click.option("--vehicle-label", default="vehicle", show_default=True, help="Label name for vehicle in stage-1 model")
@click.option("--living-label", default="living_beings", show_default=True, help="Label name for living beings in stage-1 model")
@click.option("--pad", default=4, show_default=True, type=int, help="Padding (px) around vehicle crops for stage-2")
def main(
        stage1_weights: str,
        stage2_weights: str,
        input_image: str,
        output: str,
        device: str | None,
        stage1_conf: float,
        stage2_conf: float,
        stage1_iou: float,
        stage2_iou: float,
        vehicle_label: str,
        living_label: str,
        pad: int,
) -> None:
    pipeline = TwoStagePipeline(
        stage1_weights_path=stage1_weights,
        stage2_weights_path=stage2_weights,
        vehicle_label_stage1=vehicle_label,
        living_label_stage1=living_label,
        device=device,
        stage1_conf=stage1_conf,
        stage2_conf=stage2_conf,
        stage1_iou=stage1_iou,
        stage2_iou=stage2_iou,
        crop_pad_pixels=pad,
    )
    pipeline.run_on_image(input_image, output)
    click.echo(f"Saved: {output}")


if __name__ == "__main__":
    main()
