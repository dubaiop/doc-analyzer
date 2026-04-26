import os
import tempfile
import textwrap


def _make_slide_image(title: str, body: str, slide_num: int, total: int) -> str:
    from PIL import Image, ImageDraw, ImageFont

    W, H = 1280, 720

    img = Image.new("RGB", (W, H), (14, 14, 22))
    draw = ImageDraw.Draw(img)

    for y in range(H):
        t = y / H
        draw.line([(0, y), (W, y)], fill=(int(14 + 12 * t), int(14 + 8 * t), int(22 + 20 * t)))

    draw.rectangle([(0, 0), (W, 6)], fill=(124, 58, 237))
    draw.rectangle([(0, H - 4), (W, H)], fill=(60, 30, 120))

    try:
        font_title = ImageFont.load_default(size=52)
        font_body  = ImageFont.load_default(size=28)
        font_small = ImageFont.load_default(size=20)
    except TypeError:
        font_title = font_body = font_small = ImageFont.load_default()

    draw.text((W - 100, 22), f"{slide_num} / {total}", fill=(80, 80, 120), font=font_small)

    title_lines = textwrap.wrap(title, width=38)[:2]
    for i, line in enumerate(title_lines):
        draw.text((80, 80 + i * 66), line, fill=(196, 181, 253), font=font_title)

    y_div = 80 + len(title_lines) * 66 + 16
    draw.rectangle([(80, y_div), (700, y_div + 3)], fill=(124, 58, 237))

    y = y_div + 30
    for line in textwrap.wrap(body, width=72)[:10]:
        draw.text((80, y), line, fill=(210, 210, 232), font=font_body)
        y += 46

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, format="JPEG", quality=92)
    tmp.close()
    return tmp.name


def build_video(slides: list[dict], output_path: str, lang: str = "en"):
    """
    slides: list of {"title": str, "body": str, "narration": str}
    Writes an MP4 to output_path with voice narration per slide.
    """
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
    from gtts import gTTS

    total = len(slides)
    clips = []
    tmp_files = []

    try:
        for i, slide in enumerate(slides, 1):
            img_path = _make_slide_image(slide["title"], slide["body"], i, total)
            tmp_files.append(img_path)

            tts = gTTS(text=slide["narration"], lang=lang, slow=False)
            audio_tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tts.save(audio_tmp.name)
            audio_tmp.close()
            tmp_files.append(audio_tmp.name)

            audio = AudioFileClip(audio_tmp.name)
            clip = ImageClip(img_path, duration=audio.duration + 0.8).set_audio(audio)
            clips.append(clip)

        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None,
        )
        final.close()
        for c in clips:
            try:
                c.close()
            except Exception:
                pass
    finally:
        for f in tmp_files:
            try:
                os.unlink(f)
            except Exception:
                pass
