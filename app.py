import gradio as gr

from infer import infer_image, infer_video

input_image = gr.Image(type='pil', label='Input Image')
input_model_image = gr.Radio([('x2', 2), ('x4', 4), ('x8', 8)], type="value", value=4, label="Model Upscale/Enhance Type")
submit_image_button = gr.Button('Submit')
output_image = gr.Image(type="filepath", label="Output Image")

tab_img = gr.Interface(
    fn=infer_image,
    inputs=[input_image, input_model_image],
    outputs=output_image,
    title="Image Super-Resolution and Enhancement Using Residual Channels Attention Networks for images",
)

input_video = gr.Video(label='Input Video')
input_model_video = gr.Radio([('x2', 2), ('x4', 4), ('x8', 8)], type="value", value=2, label="Model Upscale/Enhance Type")
submit_video_button = gr.Button('Submit')
output_video = gr.Video(label='Output Video')

tab_vid = gr.Interface(
    fn=infer_video,
    inputs=[input_video, input_model_video],
    outputs=output_video,
    title="Image Super-Resolution and Enhancement Using Residual Channels Attention Networks for Videos"
)

demo = gr.TabbedInterface([tab_img, tab_vid], ["Image", "Video"])

demo.launch(debug=True, show_error=True, share=True)