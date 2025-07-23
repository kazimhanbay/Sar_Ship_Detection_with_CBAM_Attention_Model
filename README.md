
<h1 align="center">Attentation Blok Mimarisi ile Geliştirilen Gemi Tespit Yöntemi</h1>

<p align="justify">
<strong>NOT:</strong> Bu çalışmada, gemi tespiti amacıyla önerilen derin öğrenme mimarisi CBM-RCNN (Channel-Boosted Multi-Scale R-CNN), temel olarak Faster R-CNN üzerine inşa edilmiştir. Modelin genel yapısı, dikkat modülleri ve çok seviyeli öznitelik birleştirme mekanizmaları ile iyileştirilmiştir. Özellikle ResNet50 omurgası üzerine entegre edilen CBAM (Convolutional Block Attention Module) blokları ile öznitelik haritalarına hem kanal hem de uzamsal dikkat uygulanarak daha odaklı temsil öğrenimi sağlanmıştır. Ayrıca, geleneksel FPN yapısı yerine BiFPN (Bidirectional Feature Pyramid Network) kullanılarak, farklı çözünürlük seviyelerindeki özniteliklerin çift yönlü bilgi akışı ile birleştirilmesi sağlanmış ve bu sayede küçük nesnelerin tespitinde de yüksek başarı elde edilmiştir.
</p>

<p align="justify">
Modelin genel akışı Şekil 1'de sunulmuştur. 
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/2d95257f-2377-4e30-a9c4-19b34c6ded29" alt="SAR Görüntüsü">
</p>

<p align="center"><strong>Şekil 1.</strong> Önerilen Attentation-based Faster R-CNN tabanlı modelin genel mimarisi.</p>


<p align="justify">
İlk olarak, giriş görüntüsü (800×800×3) 7×7 evrişim filtresi ve max pooling ile işlenerek düşük seviyeli öznitelikler çıkarılır. Ardından, ardışık ResNet katmanları (layer1-layer4) üzerinden geçirilen bu öznitelikler, sırasıyla CBAM bloklarıyla zenginleştirilir. Elde edilen çok seviyeli öznitelikler, BiFPN modülünde kanal boyutları eşitlenerek yukarıdan-aşağıya ve aşağıdan-yukarıya yönlerde ağırlıklı olarak birleştirilir. Bu işlem sonucunda oluşan öznitelik haritaları, Region Proposal Network (RPN) tarafından işlenerek olası nesne aday bölgeleri belirlenir. Daha sonra, RoI Align operasyonu ile sabit boyutta öznitelik vektörleri oluşturulur ve bu vektörler, sınıflandırma ve koordinat regresyonu amacıyla son katmanlara (box head) gönderilir. Modelin çıktısı, her tespit edilen nesne için sınıf etiketi, güven skorları ve sınırlayıcı kutulardır.
</p>

<h4>2. Gauss Fonksiyonu ile Kara/Deniz Ayrımının Yapılması (Kara İçeren Görüntülerde)</h4>
<p align="justify">
Bu çalışmada SAR görüntülerinde kara/deniz ayrımı ve gemi tespiti için Hessian matrisinden hesaplanan ikinci öz
değer matrisi kullanılmıştır. İkinci öz değer matrisi, sahil çizgisinin doğru bir şekilde belirlenmesini
sağlayan ayırt edici karakterizasyonu sunmaktadır. Bu matris, <code>σ=k</code> (genellikle <code>k=2</code>)
değeri ile Gauss filtresinden geçirilerek ortalama görüntü elde edilmiştir. Ardından, standart sapma görüntüsü,
ikinci öz değer matrisinden ortalama görüntünün çıkarılması, karesinin alınması ve tekrar Gauss filtresinden
geçirilmesiyle hesaplanmıştır.
</p>

<p align="justify">
Projede geliştirlen yönteme ait makale yazım aşamasındadır. Bu nedenle tüm açıklamalar ve diğer detaylar makalenin yayınlanması sonrasında buraya eklenecektir. 
</p>


