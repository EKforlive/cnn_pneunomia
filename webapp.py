
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
# PDX->Pneumonia Detection X-Ray)
""")
st.write(
  """
  PDX (Pneumonia Detection X-Ray) merupakan aplikasi yang memanfaatkan Artificial Intelligence yaitu domain Computer Vision yang dapat digunakan untuk mendeteksi atau diagnosis penyakit Pneumonia dari hasil X-Ray. Aplikasi ini diharapkan dapat memudahkan masyarakat umum atau tenaga medis dalam mendiagnosis pneumonia melalui hasil pemeriksaan X-Ray yang diinput.
  """
)
components.html("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js" ></script>
<div class="col-12" style="backgournd:rgb(14, 17, 23)">
  <p class="h5">Cara Penggunaan Aplikasi :</p>
  <ol>
    <li>Buka website <a href="https://pdx-ray.streamlit.app/">https://pdx-ray.streamlit.app/</a></li>
    <li>Masukkan foto / hasil gambar X-Ray pada tombol "Upload Image" yang tersedia pada website</li>
    <li>Tunggu proses berlangsung</li>
    <li>Aplikasi akan mengeluarkan hasil diagnosis penyakit sesuai dengan gambar yang diinputkan</li>
    <li>Klik tombol selesai</li>
  </ol>
</div>
""", height=300)


temp = st.file_uploader("Upload X-Ray Image")
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
st.write("""
# ABOUT US!
""")
components.html("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js" ></script>
<br>
<br>
<div class="col-12" style="backgournd:rgb(14, 17, 23)">
  <br>
  <div class="col-12 justify-content-center text-center">
    <img src="https://i.imgur.com/WIXgRHw.jpg" class="rounded-circle" height="100px" width="100"/>
    <br>
    <div class="card card-primary">
      <div class="card-body">
        <figure>
          <blockquote class="blockquote">
            <p>Setiap perjuangan pasti ada pengorbanan.</p>
          </blockquote>
          <figcaption class="blockquote-footer">
            Raditya Putra Prayoga <cite title="Source Title">Universitas Pembangunan Nasional "Veteran" Jawa Timur - Teknik Industri</cite>
          </figcaption>
        </figure>
      </div>
    </div>
  </div>
  <br>
  <div class="col-12 justify-content-center text-center">
    <img src="https://i.imgur.com/gXck5uS.jpg" class="rounded-circle" height="100px" width="100"/>
    <br>
    <div class="card card-primary">
      <div class="card-body">
        <figure>
          <blockquote class="blockquote">
            <p>Sesuatu yang membuat anda untuk menunda pekerjaan, itu bukan karena malas. Tapi karena anda menganggap hal itu sudah tidak penting.</p>
          </blockquote>
          <figcaption class="blockquote-footer">
            Firman Mulyadi <cite title="Source Title">Universitas Merdeka Madiun  - Manajemen Informatika</cite>
          </figcaption>
        </figure>
      </div>
    </div>
  </div>
  <br>
  <div class="col-12 justify-content-center text-center">
    <img src="https://i.imgur.com/5ZI4Hdy.jpg" class="rounded-circle" height="100px" width="100"/>
    <br>
    <div class="card card-primary">
      <div class="card-body">
        <figure>
          <blockquote class="blockquote">
            <p>The result is right for time and become a reality in its sweetest form.</p>
          </blockquote>
          <figcaption class="blockquote-footer">
            Emmanuela Aurelia Rachel Passa <cite title="Source Title">UUniversitas Airlangga - Teknik Biomedis</cite>
          </figcaption>
        </figure>
      </div>
    </div>
  </div>
  <br>
  <div class="col-12 justify-content-center text-center">
    <img src="https://i.imgur.com/hPhR4Xu.jpg" class="rounded-circle" height="100px" width="100"/>
    <br>
    <div class="card card-primary">
      <div class="card-body">
        <figure>
          <blockquote class="blockquote">
            <p>Kita tidak akan pernah tau hasilnya jika kita tidak pernah mencoba.</p>
          </blockquote>
          <figcaption class="blockquote-footer">
            Izati Nuramadanti <cite title="Source Title">Universitas Alma Ata Yogyakarta - Informatika</cite>
          </figcaption>
        </figure>
      </div>
    </div>
  </div>
  <br>
  <div class="col-12 justify-content-center text-center">
    <img src="https://i.imgur.com/WkvGN29.jpg" class="rounded-circle img-responsive" height="100px" width="100"/>
    <br>
    <div class="card card-primary">
      <div class="card-body">
        <figure>
          <blockquote class="blockquote">
            <p>You're on your own,kid. You always have been.</p>
          </blockquote>
          <figcaption class="blockquote-footer">
            Indah Bella Pratiwi <cite title="Source Title">Universitas Diponegoro - Oseanografi</cite>
          </figcaption>
        </figure>
      </div>
    </div>
  </div>
</div>
""", height=1500)
            

  

  
