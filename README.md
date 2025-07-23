
<h1 align="center">Attentation Blok Mimarisi ile Geliştirilen Gemi Tespit Yöntemi</h1>

<p align="justify">
123E344 nolu TÜBİTAK 1001 projesi kapsamında geliştirilen bu yöntemde dikkat mekanizması temelinde bir derin öğrenme modeli geliştirlmiştir. Gemi tespiti amacıyla önerilen derin öğrenme mimarisi CBM-RCNN (Channel-Boosted Multi-Scale R-CNN), temel olarak Faster R-CNN üzerine inşa edilmiştir. Modelin genel yapısı, dikkat modülleri ve çok seviyeli öznitelik birleştirme mekanizmaları ile iyileştirilmiştir. Özellikle ResNet50 omurgası üzerine entegre edilen CBAM (Convolutional Block Attention Module) blokları ile öznitelik haritalarına hem kanal hem de uzamsal dikkat uygulanarak daha odaklı temsil öğrenimi sağlanmıştır. Ayrıca, geleneksel FPN yapısı yerine BiFPN (Bidirectional Feature Pyramid Network) kullanılarak, farklı çözünürlük seviyelerindeki özniteliklerin çift yönlü bilgi akışı ile birleştirilmesi sağlanmış ve bu sayede küçük nesnelerin tespitinde de yüksek başarı elde edilmiştir.
</p>

<p align="justify">
Modelin genel akışı Şekil 1'de sunulmuştur. 
</p>
<img width="1161" height="821" alt="şekil 1" src="https://github.com/user-attachments/assets/6e385009-8a2b-4981-90f5-59b4390f56b6" />


<p align="center">
 <img width="800" height="450" alt="şekil 1. Önerilen Yöntemin Genel Yapısı" src="https://github.com/user-attachments/assets/6e385009-8a2b-4981-90f5-59b4390f56b6" />
</p>

<p align="center"><strong>Şekil 1.</strong> Önerilen Attentation-based Faster R-CNN tabanlı modelin genel mimarisi. </p>


<p align="justify">
İlk olarak, giriş görüntüsü (800×800×3) 7×7 evrişim filtresi ve max pooling ile işlenerek düşük seviyeli öznitelikler çıkarılır. Ardından, ardışık ResNet katmanları (layer1-layer4) üzerinden geçirilen bu öznitelikler, sırasıyla CBAM bloklarıyla zenginleştirilir. Elde edilen çok seviyeli öznitelikler, BiFPN modülünde kanal boyutları eşitlenerek yukarıdan-aşağıya ve aşağıdan-yukarıya yönlerde ağırlıklı olarak birleştirilir. Bu işlem sonucunda oluşan öznitelik haritaları, Region Proposal Network (RPN) tarafından işlenerek olası nesne aday bölgeleri belirlenir. Daha sonra, RoI Align operasyonu ile sabit boyutta öznitelik vektörleri oluşturulur ve bu vektörler, sınıflandırma ve koordinat regresyonu amacıyla son katmanlara (box head) gönderilir. Modelin çıktısı, her tespit edilen nesne için sınıf etiketi, güven skorları ve sınırlayıcı kutulardır.
</p>

<h4>1. Basit Sahnelerde Az Sayıda Gemi Tespiti</h4>
<p align="center">


</p>

<p align="justify">
<strong>NOT:Projede geliştirilen yönteme ait makale yazım aşamasındadır. Bu nedenle tüm açıklamalar ve diğer detaylar makalenin yayınlanması sonrasında buraya eklenecektir. </strong>
</p>
<p align="justify">
Geliştirilen yönteme ait bazı görsel sonuçlar aşağıda verilmiştir. 
</p>




