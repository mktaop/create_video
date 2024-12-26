
#pip install moviepy==1.0.3 --- i have to use this version for me to use moviepy.editor
#pip install google-cloud-texttospeech  --- if you have not used texttosppech before or installed

import os, json, requests, time, vertexai, fitz, streamlit as st, numpy as np
from moviepy.editor import ImageSequenceClip, AudioFileClip, ImageClip, VideoFileClip, concatenate_videoclips
from moviepy.editor import *
from PIL import Image
import PIL.Image
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
from google.cloud import texttospeech
from vertexai.preview.vision_models import ImageGenerationModel
import google.generativeai as genai
from openai import OpenAI


def setup():
    st.header("Create a video from images and audio!")
    st.sidebar.header("Options", divider='rainbow')
    

def get_llminfo():
    tip1="Select a model you want to use."
    model = st.sidebar.radio("Choose LLM:",
                                  ("gemini-1.5-flash-001",
                                   "gemini-1.5-pro-001",
                                   ), help=tip1)
    temp = 1.0 #--you can turn this into a user defined variable if you like
    topp = 0.94 #--you can turn this into a user defined variable if you like
    tip4="Number of response tokens, 8194 is limit."
    maxtokens = st.sidebar.slider("Maximum Tokens:", min_value=100,
                                  max_value=5000, value=2000, step=100, help=tip4)
    return model, temp, topp, maxtokens


def get_imgmodel():
    img_model_choice = st.sidebar.radio("Select which image model to use:",("dalle-3","imagen-3",))
    return img_model_choice


def get_example():
    exmpl = st.sidebar.radio("Choose Example:",
                             ("Narrate over PDF presentation",
                              "Create marketing ad",
                              "Create slide show from pics",)
                             )
    return exmpl


def delete_all_files(directory):
    """
    I use this function to delete files before creating new images, etc

    Parameters
    ----------
    directory : directory path where files (e.g. image) are located.

    Returns
    -------
    None.

    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        os.remove(file_path)


def count_png_files(folder_path):
    """
    This function is used to count the number for files in a directory.

    Parameters
    ----------
    folder_path : Directory path where files are located.

    Returns
    -------
    count : number of files in the directory provided.

    """
    count = 0
    for file in os.listdir(folder_path):
        if file.endswith('.png'):
            count += 1
    return count


def create_video_from_images(image_folder, audio_file, output_video_file, num_of_images, image_display_duration=2, resolution=(800, 640)):
    """
    

    Parameters
    ----------
    image_folder : TYPE
        DESCRIPTION.
    audio_file : TYPE
        DESCRIPTION.
    output_video_file : TYPE
        DESCRIPTION.
    num_of_images : TYPE
        DESCRIPTION.
    image_display_duration : TYPE, optional
        DESCRIPTION. The default is 2.
    resolution : TYPE, optional
        DESCRIPTION. The default is (800, 640).

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    #get all the PNG files in the provided folder
    image_files = [os.path.join(image_folder, img) for img in sorted(os.listdir(image_folder)) if img.endswith('.png')]
    
    if not image_files:
        raise ValueError("No PNG images found in the specified folder.")
    
    #resize each image to the desired resolution (800x640 in my case) and convert to RGB using PIL
    resized_images = []
    for image_file in image_files:
        pil_image = Image.open(image_file).convert("RGB")
        pil_image = pil_image.resize(resolution)
        #convert the PIL image to a numpy array and append it to the list
        resized_images.append(np.array(pil_image))  
    
    #load the audio clip
    audio_clip = AudioFileClip(audio_file)
    audio_duration = audio_clip.duration
    #if you want each image to appear for a certain duration then comment out the following line, and it will use the 
    #image display duration defined in the function call which is set to 2 seconds currently.
    image_display_duration = audio_duration/num_of_images 
    
    #calc the total number of times each image will be shown
    fps = 1 / image_display_duration  # Each image is displayed for `image_display_duration` seconds
    total_frames = int(audio_duration * fps)
    
    #loop through the images to create enough frames..this will matter if you have opted to show each image for 
    #a specific duration and commented out the image display duration calculation above
    #if you want to show each image for certain duration the following few lines ensure that images ared repeated as necessary
    repeated_images = []
    while len(repeated_images) < total_frames:
        repeated_images.extend(resized_images)
    repeated_images = repeated_images[:total_frames]  # Trim any extra frames
    
    #creates a video clip from the repeated images
    video_clip = ImageSequenceClip(repeated_images, fps=fps)
    
    #sets the audio for the video clip
    video_clip = video_clip.set_audio(audio_clip)
    
    #writes the final video to a file
    video_clip.write_videofile(output_video_file, codec='libx264')
    

def create_video_from_image(image_file, audio_file, output_video_file,):
    #loads the images
    image = ImageClip(image_file)

    #loads the audio
    audio = AudioFileClip(audio_file)

    #creates the video clip
    video = image.set_audio(audio).set_duration(audio.duration)

    #writes the video to a file
    video.write_videofile(output_video_file, fps=24)
    

def concatenate_videos(folder_path, output_file):
    """
    This function will take multiple video files and concatenate them into one video.

    Parameters
    ----------
    folder_path : Folder where you have the video files stored.
    output_file : Name and location for the final concatenated video file.

    Returns
    -------
    None.

    """
    video_clips = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.mp4', '.avi', '.mov')):  # Add more extensions if needed
            filepath = os.path.join(folder_path, filename)
            video_clips.append(VideoFileClip(filepath))

    final_clip = concatenate_videoclips(video_clips)
    final_clip.write_videofile(output_file)
    
    
def main():
    """
    set up the front end
    give options for model, tokens, etc
    3 examples: 1.) ask gemini to give us some images and a script we can use to create a video or an ad
                2.) take pictures to create a slideshow and overlay with audio
                3.) slice up a ppt and have gemini give narration for each slide and create that into audio and create video
    for each example:
        1.) ask user for the prompt 
        2.) get the response and use the image prompts to retrieve images form imagen-3
        3.) stitch the images and overlay narration or music
        4.) create video and play

    Returns
    -------
    None.

    """
    setup()
    model_select, temperature, top_p,  max_tokens = get_llminfo()
    
    example = get_example()
    
    generation_config = {
      "temperature": temperature,
      "top_p": top_p,
      "max_output_tokens": max_tokens,
      "response_mime_type": "application/json",
      }
    model = genai.GenerativeModel(
      model_name=model_select,
      generation_config=generation_config,)
    
    if example == "Create marketing ad":
        img_model_choice = get_imgmodel()
        numimages = st.text_input('How many images?')  
        if not numimages: st.stop()                                                            
        category = st.text_input('Name of the category?')
        if not category: st.stop()
        product = st.text_input('Name of the product?') 
        if not product: st.stop()
        text1 = """Give me 3 different image descriptions that i will use to generate images with and then i will use these images to create a video for {product} advertisement 
                   featuring the {category}.  Do not include person/human in the image description.
                   also give me a script or narration that i can use in the video, 
                   make this no more than 60 words. final response should be in json format 
                   where keys are [\'scenes\', \'script\']  """
        final_text = text1.format(numimages=numimages, category=category, product=product)
        
        st.write("The following pre-written prompt is sent to the LLM to generate image descriptions \\ which are used as prompt into the vision model.")
        st.markdown(final_text)

        response = model.generate_content([final_text], stream=False)
        json_string = f"""
        {response.text}
        """
        data = json.loads(json_string)
        num_of_images = len(data['scenes'])

        delete_all_files("/Users/avi_patel/Documents/ytdemo")
        
        for numx2 in range(num_of_images):
            img_prompt = data['scenes'][numx2]
            st.write(f"creating image {numx2}")
            if img_model_choice == "dalle-3":
                response = oaiclient.images.generate(
                  model="dall-e-3",
                  prompt = img_prompt,
                  n=1,
                )
                img_data = requests.get(response.data[0].url, stream=True).content
                outputpath = f"/Users/avi_patel/Documents/ytdemo/scene_{numx2}.png"
                with open(outputpath, 'wb') as handler:
                    handler.write(img_data)
            elif img_model_choice == "imagen-3":
                imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
                response = imagen_model.generate_images(prompt=img_prompt, number_of_images=1)
                response.images[0].save(location=f"/Users/avi_patel/Documents/ytdemo/scene_{numx2}.png")
        
        st.write("creating audio narration")
        script = data['script']
        voice_model = "en-US-Standard-B" #for female voice use "en-US-Standard-A"
        input_text = texttospeech.SynthesisInput(text=script)
        voice = texttospeech.VoiceSelectionParams(language_code="en-US", name=voice_model)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16, speaking_rate=1)

        speech_client = texttospeech.TextToSpeechClient()
        response = speech_client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )

        audio_output = "/Users/avi_patel/Documents/ytdemo/narration.mp3"
        with open(audio_output, "wb") as out:
            out.write(response.audio_content)
        
        image_folder = "/Users/avi_patel/Documents/ytdemo"  # Folder where the PNG images are stored
        audio_file = audio_output # Path to the audio file (can be .mp3, .wav, etc. in my case .mp3)
        output_video_file = "/Users/avi_patel/Documents/ytdemo/output_video.mp4"  # Output video file name
        st.write("creating the video")
        create_video_from_images(image_folder, audio_file, output_video_file, num_of_images, image_display_duration=2)
        st.video(output_video_file)
                
    elif example == "Create slide show from pics":
        pathtofiles = st.text_input("Provide path to your pictures you want to convert into a slideshow!")
        if not pathtofiles: st.stop()
        pathtoaudio = st.text_input("Provide path and filename to the audio file you want to overlay on the slideshow!")
        if not pathtoaudio: st.stop()
        outputfile = st.text_input("Provide a filename and path to save the final video created with your pictures and audio!")
        if not outputfile: st.stop()
        image_folder = pathtofiles
        audio_file = pathtoaudio
        output_video_file = outputfile
        png_count = count_png_files(pathtofiles)
        create_video_from_images(image_folder, audio_file, output_video_file, png_count, image_display_duration=2)
        st.video(output_video_file)
        
    elif example == "Narrate over PDF presentation":
        pathtopdf = st.text_input("Provide path and filename to your PDF presentation.")
        if not pathtopdf: st.stop()
        file_path = pathtopdf
        delete_all_files("/Users/avi_patel/Documents/ytdemo3")
        doc = fitz.open(file_path)  # open document
        for i, page in enumerate(doc):
            pix = page.get_pixmap()  # render page to an image
            pix.save(f"/Users/avi_patel/Documents/ytdemo3/page_{i}.png")
            
        png_count = count_png_files("/Users/avi_patel/Documents/ytdemo3")
        generation_config = {
          "temperature": temperature,
          "top_p": top_p,
          "max_output_tokens": max_tokens,}
        model2 = genai.GenerativeModel(model_name=model_select, generation_config=generation_config,)
        prompt2 = st.text_input("Enter your prompt.") 
        if not prompt2: st.stop()
        for count in range(png_count):
            file = f"/Users/avi_patel/Documents/ytdemo3/page_{count}.png"
            image_file = genai.upload_file(path=file)
            
            while image_file.state.name == "PROCESSING":
                time.sleep(10)
                image_file = genai.get_file(image_file.name)
            if image_file.state.name == "FAILED":
              raise ValueError(image_file.state.name)
            
            response = model2.generate_content([image_file, prompt2],
                                              request_options={"timeout": 600})
            narration = response.text
            voice_model = "en-US-Standard-A" #for male voice use "en-US-Standard-B"
            input_text = texttospeech.SynthesisInput(text=narration)
            voice = texttospeech.VoiceSelectionParams(language_code="en-US", name=voice_model)
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16, speaking_rate=1)

            speech_client = texttospeech.TextToSpeechClient()
            response = speech_client.synthesize_speech(
                request={"input": input_text, "voice": voice, "audio_config": audio_config}
            )

            audio_output = f"/Users/avi_patel/Documents/ytdemo3/narration_{count}.mp3"
            with open(audio_output, "wb") as out:
                out.write(response.audio_content)
            video_output = f"/Users/avi_patel/Documents/ytdemo3/video_{count}.mp4"
            create_video_from_image(file, audio_output, video_output)
                             
        folder_path = "/Users/avi_patel/Documents/ytdemo3"
        output_file = "/Users/avi_patel/Documents/ytdemo3/finalclip.mp4"
        concatenate_videos(folder_path, output_file)
        st.video(output_file)
        

if __name__ == "__main__":
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY_NEW')
    genai.configure(api_key=GOOGLE_API_KEY)
    projectid = os.environ.get('GOOG_PROJECT')
    vertexai.init(project=projectid, location="us-central1")
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    oaiclient = OpenAI(api_key=OPENAI_API_KEY)
    main()
