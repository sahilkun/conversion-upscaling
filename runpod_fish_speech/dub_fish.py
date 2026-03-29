#!/usr/bin/env python3
"""Minimal Fish Speech dubbing script - uses pre-separated audio."""
import os, sys, re, subprocess, requests, ormsgpack
sys.path.insert(0, '/workspace/fish-speech')
from tools.schema import ServeTTSRequest, ServeReferenceAudio

API = "http://127.0.0.1:8080/v1/tts"
WORK = "/workspace/dub_work"
VOCALS = f"{WORK}/htdemucs/input/vocals.wav"
BG = f"{WORK}/htdemucs/input/no_vocals.wav"
REF = f"{WORK}/refs/female1.wav"
CLIPS = f"{WORK}/clips"
SUBS = "/workspace/subtitle.ass"
AUDIO = "/workspace/audio_japanese.mp4"
OUTPUT = "/workspace/english_dub.wav"

def parse_ass():
    dialogues = []
    for line in open(SUBS, encoding="utf-8-sig"):
        if not line.startswith("Dialogue:"): continue
        p = line.split(",", 9)
        if len(p) < 10: continue
        style = p[3].strip()
        if style.lower() in ("sign","signs","credits","credit","note"): continue
        def ts(t):
            h,m,s = t.strip().split(":")
            return int(h)*3600+int(m)*60+float(s)
        start, end = ts(p[1]), ts(p[2])
        text = re.sub(r"\{[^}]*\}", "", p[9].strip()).replace("\\N"," ").replace("\\n"," ").strip()
        if not text or len(text) < 2: continue
        dialogues.append({"start":start,"end":end,"dur":end-start,"text":text})
    dialogues.sort(key=lambda d: d["start"])
    return dialogues

def tts(text, out):
    ref = open(REF, "rb").read()
    req = ServeTTSRequest(
        text=text,
        references=[ServeReferenceAudio(audio=ref, text="")],
        format="wav", streaming=False,
    )
    r = requests.post(API,
        data=ormsgpack.packb(req, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
        headers={"content-type": "application/msgpack"}, timeout=120)
    if r.status_code == 200 and len(r.content) > 1000:
        open(out, "wb").write(r.content)
        return True
    return False

# Get duration
dur = float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
    "-of","default=noprint_wrappers=1:nokey=1",AUDIO],capture_output=True,text=True).stdout.strip())
print(f"Duration: {dur:.0f}s")

# Parse
dl = parse_ass()
print(f"Parsed {len(dl)} lines")

# Generate TTS
os.makedirs(CLIPS, exist_ok=True)
manifest = []
for i, d in enumerate(dl):
    clip = f"{CLIPS}/line_{i:04d}.wav"
    if d["dur"] < 0.3: continue
    if tts(d["text"], clip):
        manifest.append({"path":clip,"start":d["start"],"dur":d["dur"]})
    else:
        print(f"  FAIL {i}: {d['text'][:40]}")
    if (i+1) % 25 == 0:
        print(f"  {i+1}/{len(dl)} ({len(manifest)} ok)")
print(f"Generated {len(manifest)} clips")

if not manifest:
    print("ERROR: No clips generated!")
    sys.exit(1)

# Mix voice track
voice = f"{WORK}/voice_only.wav"
ins = ["-f","lavfi","-i",f"anullsrc=r=44100:cl=mono:d={dur}"]
fp = []
for j,m in enumerate(manifest):
    ins.extend(["-i",m["path"]])
    ms = int(m["start"]*1000)
    try:
        cd = float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
            "-of","default=noprint_wrappers=1:nokey=1",m["path"]],capture_output=True,text=True).stdout.strip())
    except: cd = m["dur"]
    r = cd/m["dur"] if m["dur"]>0 else 1
    if r > 1.1:
        fp.append(f"[{j+1}]atempo={min(r,2.0):.3f},adelay={ms}|{ms}[d{j}]")
    else:
        fp.append(f"[{j+1}]adelay={ms}|{ms}[d{j}]")
mi = "[0]"+"".join(f"[d{j}]" for j in range(len(manifest)))
fp.append(f"{mi}amix=inputs={len(manifest)+1}:duration=first:dropout_transition=0[voice]")
print("Mixing voice track...")
subprocess.run(["ffmpeg","-y"]+ins+["-filter_complex",";".join(fp),"-map","[voice]",
    "-acodec","pcm_s16le","-ar","44100","-ac","1",voice],check=True,capture_output=True)

# Duck + final mix
print("Final mix...")
pts = [(m["start"],m["start"]+m["dur"]) for m in manifest]
ex = "+".join(f"between(t,{s:.2f},{e:.2f})" for s,e in pts)
vf = f"volume='if({ex},0.15,0.8)':eval=frame"
subprocess.run(["ffmpeg","-y","-i",voice,"-i",VOCALS,"-i",BG,"-filter_complex",
    f"[0]volume=1.8[en];[1]{vf}[ja];[2]volume=0.7[bg];[en][ja][bg]amix=inputs=3:duration=longest[out]",
    "-map","[out]","-acodec","pcm_s16le","-ar","44100","-ac","2",OUTPUT],check=True,capture_output=True)

sz = os.path.getsize(OUTPUT)/(1024*1024)
print(f"\nDONE! {OUTPUT} ({sz:.1f} MB)")
