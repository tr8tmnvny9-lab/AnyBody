import asyncio
import subprocess
import wave
from pathlib import Path
import edge_tts
import imageio_ffmpeg

VOICE_DIR = Path("target/voice_cache_v2")
VOICE_DIR.mkdir(parents=True, exist_ok=True)

V2_SCRIPT = [
    ("vo_01_holy_grail", 1.10, "For decades, the Holy Grail of robotics has been a single brain that can understand any human command and physically execute it on a full-scale humanoid."),
    ("vo_02_anybody_bridge", 9.00, "Meet AnyBody: the first body-agnostic, task-agnostic system uniting cloud L L M and vision models with human-like spiking neurons to control humanoids and robotics end-to-end."),
    ("vo_03_arch_connectome", 23.40, "AnyBody is both task-agnostic and body-agnostic. Given only basic joint and link metadata, its twenty-five-million-synapse Rust connectome self-sorts its motor map—already tested across humanoids, dexterous hands, and quadruped robot dogs."),
    ("vo_04_arch_meta_models", 37.50, "To make it task-agnostic, Meta Muse Spark, voice input, and SAM three point one agentically decompose open-ended goals and first-person vision into spatial attractors that drive human-like spiking neurons."),
    ("vo_05_arch_neurons_cb", 50.00, "Inside the connectome, two-compartment Layer-Five pyramidal neurons multiplex top-down intent with bottom-up proprioception, while a forty-four millisecond cerebellar eligibility trace cancels thirty milliseconds of sensory delay."),
    ("vo_06_arch_wattage", 63.00, "Down in the spinal cord, half-center rhythm generators and Hill-type muscles stabilize the body at four hundred twenty-two Watts—nearly three times more efficient than stiff P I D control."),
    ("vo_07_demo1_squat", 74.50, "Watch Demo One execute a five-stage compound prompt zero-shot. Unitree H1 walks across the floor to the red box, drops into a fifty-one centimeter deep squat with its feet anchored at zero height, and grasps the box via tactile slip reflexes."),
    ("vo_08_demo1_lift", 90.50, "As basal ganglia disinhibition gates each stage automatically, H1 stands, pivots one hundred fifty degrees left, and presses the box to one point three one meters overhead—with zero keyframe animations."),
    ("vo_09_demo2_kick", 104.00, "Demo Two stress-tests single-leg vestibular balance. Commanded to kick high, the cerebellar forward model shifts H1's center of mass over the left foot, holding roll error under one point two degrees while the right leg swings to forty-seven centimeters."),
    ("vo_10_demo2_bow_wave", 119.80, "Without resetting or losing ground contact, the connectome flows directly into a voluntary seventeen-degree forward bow and a phase-locked bimanual wave."),
    ("vo_11_demo3_agentic", 130.50, "Finally, Demo Three unites our Live Generative Agentic U I with H1's actual first-person SAM three point one camera feed. Given one high-level goal—clean all the things in the basket—H1 autonomously chains perception and whole-body action."),
    ("vo_12_demo3_audit", 147.00, "Each time H1 delivers an object and rotates its head, SAM three point one segments the next unseen item in real time, spawning new task cards until all four objects sit inside the basket and a three-sixty visual audit halts the loop."),
    ("vo_13_outro_baseten", 165.20, "Architected from first principles in Finland by Oscar Kinnala, with special thanks to Baseten for sponsoring one hundred euros in inference compute powering our live Meta Muse and SAM three point one pipeline. Thank you for watching!"),
]

async def main():
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # Force regenerate vo_03 and vo_04 with the body-agnostic & task-agnostic script
    for regen_id in ["vo_02_anybody_bridge"]:
        for ext in [".mp3", ".wav"]:
            p = VOICE_DIR / f"{regen_id}{ext}"
            if p.exists():
                p.unlink()

    prev_end = 0.0
    for vid, nom_start, text in V2_SCRIPT:
        mp3_p = VOICE_DIR / f"{vid}.mp3"
        wav_p = VOICE_DIR / f"{vid}.wav"
        if not wav_p.exists():
            comm = edge_tts.Communicate(text, voice="en-US-AndrewNeural", rate="+9%", pitch="-1Hz")
            await comm.save(str(mp3_p))
            subprocess.run([ffmpeg, "-y", "-i", str(mp3_p), "-ar", "48000", "-ac", "1", str(wav_p)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with wave.open(str(wav_p), "rb") as wf:
            dur = wf.getnframes() / float(wf.getframerate())
        actual_start = max(nom_start, round(prev_end + 0.65, 2))
        actual_end = round(actual_start + dur, 2)
        gap = actual_start - prev_end if prev_end > 0 else actual_start
        prev_end = actual_end
        print(f"{vid:22s} | start={actual_start:6.2f}s | dur={dur:5.2f}s | end={actual_end:6.2f}s | gap_before={gap:4.2f}s")

asyncio.run(main())
