
import streamlit as st
import numpy as np
from PIL import Image 
from tensorflow.keras.models import load_model
import tensorflow as tf
 
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image 
import streamlit.components.v1 as components

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

def loading_model():
  fp = "cnn_model.h5"
  model_loader = load_model(fp)
  return model_loader

cnn = loading_model()
st.write("""
# PDX (Pneumonia Detection X-Ray)
""")
st.write(
  """
  PDX (Pneumonia Detection X-Ray) merupakan aplikasi yang memanfaatkan Artificial Intelligence yaitu domain Computer Vision yang dapat digunakan untuk mendeteksi atau diagnosis penyakit Pneumonia dari hasil X-Ray. Aplikasi ini diharapkan dapat memudahkan masyarakat umum atau tenaga medis dalam mendiagnosis pneumonia melalui hasil pemeriksaan X-Ray yang diinput.
  """
)
st.write("""
# Cara Penggunaan
""")
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js" ></script>
<hr>
<div class="col-12">
  <p class="h5">Cara Penggunaan Aplikasi :</p>
  <ol>
    <li>Buka website https://pdx-ray.streamlit.app/</li>
    <li>Masukkan foto / hasil gambar X-Ray pada tombol "Upload Image" yang tersedia pada website</li>
    <li>Tunggu proses berlangsung</li>
    <li>Aplikasi akan mengeluarkan hasil diagnosis penyakit sesuai dengan gambar yang diinputkan</li>
    <li>Klik tombol selesai</li>
  </ol>
</div>
<hr>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; '>Upload X-Ray Image</h1>", unsafe_allow_html=True)
temp = st.file_uploader("")
#temp = temp.decode()

buffer = temp
temp_file = NamedTemporaryFile(delete=False)
if buffer:
    temp_file.write(buffer.getvalue())
    st.write(image.load_img(temp_file.name))


if buffer is None:
  st.text("Oops! that doesn't look like an image. Try again.")

else:

 

  hardik_img = image.load_img(temp_file.name, target_size=(500, 500),color_mode='grayscale')

  # Preprocessing the image
  pp_hardik_img = image.img_to_array(hardik_img)
  pp_hardik_img = pp_hardik_img/255
  pp_hardik_img = np.expand_dims(pp_hardik_img, axis=0)

  #predict
  hardik_preds= cnn.predict(pp_hardik_img)
  if hardik_preds>= 0.5:
    out = ('I am {:.2%} percent confirmed that this is a Pneumonia case'.format(hardik_preds[0][0]))
  
  else: 
    out = ('I am {:.2%} percent confirmed that this is a Normal case'.format(1-hardik_preds[0][0]))

  st.success(out)
  
  image = Image.open(temp)
  st.image(image,use_column_width=True)
  st.write('')
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js" ></script>
<hr>
<p class="text-center h1">ABOUT US</p>
<div class="col-12" style="backgournd:rgb(14, 17, 23)">
  <div class="row justify-content-center">
    <div class="col-5 col-sm-12 col-lg-5 col-xl-5 mt-4">
      <div class="col-12 justify-content-center text-center">
        <div class="card bg-primary">
          <div class="card-header">
            <img src="https://i.imgur.com/WIXgRHw.jpg" class="rounded-circle" height="100px" width="100"/>
          </div>
          <div class="card-body">
            <figure>
              <blockquote class="blockquote">
                <p>Setiap perjuangan pasti ada pengorbanan.</p>
              </blockquote>
              <figcaption class="blockquote-footer text-white">
                Raditya Putra Prayoga <cite title="Source Title">Universitas Pembangunan Nasional "Veteran" Jawa Timur - Teknik Industri</cite>
              </figcaption>
            </figure>
          </div>
        </div>
      </div>
    </div>
    <div class="col-5 col-sm-12 col-lg-5 col-xl-5 mt-4">
      <div class="col-12 justify-content-center text-center">
        <div class="card bg-danger">
          <div class="card-header">
            <img src="https://i.imgur.com/gXck5uS.jpg" class="rounded-circle" height="100px" width="100"/>
          </div>
          <div class="card-body">
            <figure>
              <blockquote class="blockquote">
                <p>Sesuatu yang membuat anda untuk menunda pekerjaan, itu bukan karena malas. Tapi karena anda menganggap hal itu sudah tidak penting.</p>
              </blockquote>
              <figcaption class="blockquote-footer text-white">
                Firman Mulyadi <cite title="Source Title">Universitas Merdeka Madiun  - Manajemen Informatika</cite>
              </figcaption>
            </figure>
          </div>
        </div>
      </div>
    </div>
    <div class="col-5 col-sm-12 col-lg-5 col-xl-5 mt-4">
      <div class="col-12 justify-content-center text-center">
        <div class="card bg-danger">
          <div class="card-header">
            <img src="https://i.imgur.com/5ZI4Hdy.jpg" class="rounded-circle" height="100px" width="100"/>
          </div>
          <div class="card-body">
            <figure>
              <blockquote class="blockquote">
                <p>The result is right for time and become a reality in its sweetest form.</p>
              </blockquote>
              <figcaption class="blockquote-footer text-white">
                Emmanuela Aurelia Rachel Passa <cite title="Source Title">UUniversitas Airlangga - Teknik Biomedis</cite>
              </figcaption>
            </figure>
          </div>
        </div>
      </div>
    </div>
    <div class="col-5 col-sm-12 col-lg-5 col-xl-5 mt-4">
      <div class="col-12 justify-content-center text-center">
        <div class="card bg-primary">
          <div class="card-header">
            <img src="https://i.imgur.com/hPhR4Xu.jpg" class="rounded-circle" height="100px" width="100"/>
          </div>
          <div class="card-body">
            <figure>
              <blockquote class="blockquote">
                <p>Kita tidak akan pernah tau hasilnya jika kita tidak pernah mencoba.</p>
              </blockquote>
              <figcaption class="blockquote-footer text-white">
                Izati Nuramadanti <cite title="Source Title">Universitas Alma Ata Yogyakarta - Informatika</cite>
              </figcaption>
            </figure>
          </div>
        </div>
      </div>
    </div>
    <div class="col-5 col-sm-12 col-lg-5 col-xl-5 mt-4">
      <div class="col-12 justify-content-center text-center">
        <div class="card bg-primary">
          <div class="card-header">
            <img src="https://i.imgur.com/WkvGN29.jpg" class="rounded-circle" height="100px" width="100"/>
          </div>
          <div class="card-body">
            <figure>
              <blockquote class="blockquote">
                <p>You're on your own,kid. You always have been.</p>
              </blockquote>
              <figcaption class="blockquote-footer text-white">
                Indah Bella Pratiwi <cite title="Source Title">Universitas Diponegoro - Oseanografi</cite>
              </figcaption>
            </figure>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
            

  

  
