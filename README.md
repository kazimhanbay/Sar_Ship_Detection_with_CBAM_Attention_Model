
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
<p align="center">Geliştirilen yönteme ait bazı görsel sonuçlar aşağıda verilmiştir. </p>

<h4>1. Basit Sahnelerde Az Sayıda Gemi Tespiti</h4>
<p align="center">
<img width="1609" height="1108" alt="şekil 2. CBM-RCNN modelinin tespit sonuçları" src="https://github.com/user-attachments/assets/ffa1c074-41d4-45db-913e-b479b0ef8644"/>
</p>
<p align="center"><strong>Şekil 2.</strong> Az sayıda gemi içeren basit deniz sahnelerinde CBM-RCNN modelinin tespit sonuçları. Sol sütunlarda ground truth kutuları (kırmızı), sağ sütunlarda ise modelin tahmin ettiği gemi merkezleri ve olasılık skorları gösterilmiştir.  </p>


<p align="justify">
<h4>2. Yoğun Gemi Trafiği Bulunan Karmaşık Sahnelerde Tespit Performansı</h4>
</p>

<p align="center">
<img width="1551" height="1097" alt="şekil 3. CBM-RCNN modelinin tespit sonuçları" src="https://github.com/user-attachments/assets/c379ae68-6ef5-492e-a669-7153f5fc0740"/>
</p>
<p align="center"><strong>Şekil 3.</strong> Kara ve deniz alanlarını aynı anda içeren, ancak düşük gemi yoğunluğuna sahip sade sahnelerde modelin tespit sonuçları. Sol sütunda ground truth anotasyonları, sağ sütunda ise CBM-RCNN tahminleri yer almaktadır. Görsellerde, sahil hattı boyunca konumlanmış gemiler başarılı şekilde tespit edilmiştir. </p>

<p align="justify">
<h4>3. Kara ve Deniz Unsurlarının Birlikte Bulunduğu Basit Sahnelerde Tespit Performansı</h4>
</p>
<p align="center">
<img width="1400" height="800" alt="şekil 4. CBM-RCNN modelinin tespit sonuçları" src="https://github.com/user-attachments/assets/cc751d74-c309-4936-9ae3-d6e2a291a358"/>
</p>
<p align="center"><strong>Şekil 4.</strong> Kara ve deniz alanlarını aynı anda içeren, ancak düşük gemi yoğunluğuna sahip sade sahnelerde modelin tespit sonuçları. Sol sütunda ground truth anotasyonları, sağ sütunda ise CBM-RCNN tahminleri yer almaktadır. Görsellerde, sahil hattı boyunca konumlanmış gemiler başarılı şekilde tespit edilmiştir.  </p>




<p align="justify">
<strong>NOT:Projede geliştirilen yönteme ait makale yazım aşamasındadır. Bu nedenle tüm açıklamalar ve diğer detaylar makalenin yayınlanması sonrasında buraya eklenecektir. </strong>
</p>





