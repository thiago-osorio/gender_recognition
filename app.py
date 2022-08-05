import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
import streamlit.components.v1 as components
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures

def specprop(filename):
    '''
    Function to get the acoustic and speech properties of the voice
    '''
    
    [fs, data] = audioBasicIO.read_audio_file(filename)

    if data.ndim > 1:
        data = data[:, 0]

    spec = np.abs(np.fft.rfft(data))
    freq = np.fft.rfftfreq(len(data), d=1/fs)

    assert len(spec) == len(freq)

    amp = spec / spec.sum()
    amp_cumsum = amp.cumsum()

    assert len(amp_cumsum) == len(freq)

    mean = (freq * amp).sum()
    sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
    median = freq[len(amp_cumsum[amp_cumsum <= 0.5])]
    mode = freq[amp.argmax()]
    q25 = freq[len(amp_cumsum[amp_cumsum < 0.25])]
    q75 = freq[len(amp_cumsum[amp_cumsum < 0.75])]
    z = amp - amp.mean()
    w = amp.std()
    iqr = q75 - q25
    skew = ((z ** 3).sum() / (len(spec) - 1)) / w ** 3
    kurt = ((z ** 4).sum() / (len(spec) - 1)) / w ** 4
    
    F, f_names = ShortTermFeatures.feature_extraction(data, fs, 0.050*fs, 0.025*fs)
    props = [
        {'name': 'mean', 'value': mean},
        {'name': 'sd', 'value': sd},
        {'name': 'median', 'value': median},
        {'name': 'mode', 'value': mode},
        {'name': 'q25', 'value': q25}, 
        {'name': 'q75', 'value': q75},
        {'name': 'iqr', 'value': iqr},
        {'name': 'skew', 'value': skew},
        {'name': 'kurt', 'value': kurt}
        ]
    for i in range(8):
        props.append({
            'name': f_names[i],
            'value': F[i].mean()
        })
        
    props = pd.DataFrame(props)
    props = props.T
    props.reset_index(inplace=True, drop=True)
    props.columns = ['mean', 'sd', 'median', 'mode', 'q25', 'q75', 'iqr', 'skew', 'kurt', 'zcr', 'energy', 'energy_entropy', 'spectral_centroid', 'spectral_spread', 'spectral_entropy', 'spectral_flux', 'spectral_rolloff']
    props.drop(0, axis=0, inplace=True)
    
    return props
def predicao(modelo, features):
    pred = modelo.predict(features)
    if pred == 1:
            return 'Male'
    elif pred == 0:
        return 'Female'
    else:
        return 'Deu ruim bro'

parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
st_audiorec = components.declare_component("st_audiorec", path=build_dir)

st.set_page_config(page_title="Gender Recognition by Voice")

st.title('Gender Recognition')

val = st_audiorec()

if isinstance(val, dict):
    with st.spinner('Processing audio-recording...'):
        ind, val = zip(*val['arr'].items())
        ind = np.array(ind, dtype=int)
        val = np.array(val)
        sorted_ints = val[ind]
        stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
        wav_bytes = stream.read()
        with open('temp//audio.wav', 'wb') as f:
            f.write(wav_bytes)
            f.close()
        properties = specprop('temp/audio.wav')
        modelo = pickle.load(open('modelos/pipeline.pkl', 'rb'))
        pred = predicao(modelo, properties)
        if pred == 'Male' or pred == 'Female':
            if pred == 'Male':
                st.success(pred)
                st.image('https://cdn-icons-png.flaticon.com/512/44/44483.png', width=60)
            else:
                st.success(pred)
                st.image('https://www.iconpacks.net/icons/2/free-female-symbol-icon-2240-thumb.png', width=60)
        else:
            st.error(pred)