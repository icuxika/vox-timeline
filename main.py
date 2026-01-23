import os
import argparse
from src.pipeline.dubbing import VideoDubber

def main():
    parser = argparse.ArgumentParser(description="Vox Timeline Dubbing Tool")
    parser.add_argument("--script", type=str, default="demo_script.json", help="Path to the dubbing script (JSON)")
    parser.add_argument("--output", type=str, default="output_audio.wav", help="Path to the output audio file")
    parser.add_argument("--debug-dir", type=str, help="Optional: Directory to save individual audio segments for debugging")
    parser.add_argument("--video", type=str, help="Optional: Path to video file to dub")
    parser.add_argument("--video-out", type=str, default="output_video.mp4", help="Output video path if video is provided")
    
    args = parser.parse_args()

    dubber = VideoDubber()
    
    # 1. Load Script
    if not os.path.exists(args.script):
        print(f"Script file not found: {args.script}")
        # Create a dummy one if it doesn't exist? No, better to warn.
        return

    script = dubber.load_script(args.script)
    
    # 2. Generate Audio
    dubber.generate_audio_track(script, args.output, debug_dir=args.debug_dir)
    
    # 3. (Optional) Dub Video
    if args.video:
        if not os.path.exists(args.video):
            print(f"Video file not found: {args.video}")
            return
        
        dubber.dub_video(args.video, args.output, args.video_out)

if __name__ == "__main__":
    main()
