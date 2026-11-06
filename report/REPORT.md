# REPORT — Ngày 1: Đọc prediction YOLO11 để hiểu định dạng nhãn

---

## 1. Phân loại ảnh

### 1.1 Chép record hạng 1

> - `class_id` = **468**
> - `class_name` = **cab**
> - `rank` = **1**
> - `score` = **0.510915**
> - `taxonomy_name` = **ImageNet-1K**

### 1.2 Record này nói về cả bức ảnh hay một vật thể riêng? Vì sao?

> Record nói về **cả bức ảnh**. Record chỉ có `sample_id`, `image_width`, `image_height` và `rank`, không có `bbox_xyxy` hay `polygon_xy` để chỉ vị trí một vật thể. Task là `image_classification`, tức là một lớp cho toàn bộ ảnh. Ảnh `traffic` có nhiều xe buýt, ô tô và người, nhưng model chỉ trả lời chung là "cả ảnh giống `cab` nhất".

### 1.3 Danh sách tên lớp do đâu mà có?

> Model **không tự nghĩ ra** tên lớp. Danh sách lớp đi kèm **taxonomy/checkpoint**: file `yolo11n-cls.pt` được huấn luyện sẵn với taxonomy ImageNet-1K (1000 lớp cố định), và `class_id` chỉ là chỉ số trong danh sách đó. Vì vậy model chỉ chọn được trong 1000 lớp này. Ví dụ ảnh `kitchen` bị gán `gong` (score 0.42) vì ImageNet-1K không có lớp "kitchen" phù hợp.

### 1.4 Vì sao ghi cả `class_id`, `class_name` và `taxonomy_name`?

> - `class_id` là mã số mà máy và code dùng, ngắn gọn và không mơ hồ.
> - `class_name` giúp người đọc hiểu ngay.
> - `taxonomy_name` cho biết số đó thuộc danh sách nào. Cùng `class_id` nhưng ở taxonomy khác (ví dụ COCO-80 chỉ có 80 lớp) thì có thể là lớp hoàn toàn khác hoặc không tồn tại.
>
> Nếu chỉ ghi tên lớp thì dễ nhầm giữa các tên gần giống (ví dụ `restaurant` và `dining_table`), dễ lệch khi đổi taxonomy và khó ghép lại với model. Ghi đủ cả ba thì dữ liệu mới tái lập và kiểm tra được.

### 1.5 Ảnh có nhiều vật hoặc nhiều chủ đề thì guideline cần nói rõ gì?

> Guideline phải nêu **quy tắc chọn đúng một lớp**, ví dụ:
> - Chọn theo chủ thể chính hay vật chiếm diện tích lớn nhất.
> - Hoặc chọn theo bối cảnh chung của cảnh (ảnh bàn ăn nhiều đồ vật thì chọn `restaurant`).
> - Cách xử lý khi hai chủ đề ngang nhau.
> - Ca khó, mơ hồ thì **escalation**: hỏi Lab Coach, không tự bịa nhãn.
>
> Không có quy tắc này, hai annotator sẽ chọn hai nhãn khác nhau cho cùng một ảnh, làm ground truth thiếu nhất quán.

### 1.6 Vì sao không được lấy `score` của model làm ground truth?

> - `score` là **độ tin cậy của model**, không phải độ đúng của nhãn. Model có thể sai mà vẫn tự tin: `kitchen` bị gán `gong` với 0.42, `traffic` được gán `cab` với 0.51 dù ảnh có nhiều xe buýt.
> - Prediction có thể sai vì taxonomy không có lớp phù hợp.
> - Ground truth phải do **người gán nhãn theo guideline** tạo ra và được reviewer kiểm tra (QC). Nếu lấy output của model làm đáp án rồi dùng nó để đánh giá chính model đó thì kết quả bị vòng lặp, không có ý nghĩa.

---

## 2. Phát hiện vật thể

### 2.1 Chép một record

> - `class_name` = **person**
> - `score` = **0.912625**
> - `bbox_xyxy` = **[385.33, 69.24, 498.92, 348.92]**
> - `bbox_width` = **113.58**
> - `bbox_height` = **279.68**

### 2.2 Giải thích box bằng lời thường

> - **Góc trên-trái** là điểm `(385.33, 69.24)`: cách mép trái ảnh 385.33 pixel và cách mép trên ảnh 69.24 pixel.
> - **Góc dưới-phải** là điểm `(498.92, 348.92)`: cách mép trái 498.92 pixel và cách mép trên 348.92 pixel.
> - Hộp **rộng khoảng 113.58 pixel** và **cao khoảng 279.68 pixel**, là hình chữ nhật đứng, vừa khít người đàn ông mặc tạp dề đứng quay lưng.
> - Gốc tọa độ `(0, 0)` ở **góc trên-trái** ảnh; x tăng sang phải, y tăng xuống dưới. Ảnh có kích thước 640 x 427 pixel.

### 2.3 Hai ngưỡng và số prediction

> | Ngưỡng `conf` | Số prediction (ảnh `kitchen`) |
> | --- | --- |
> | **0.20** | **17** |
> | 0.35 (ngưỡng của JSON bài lab) | 11 |
> | **0.60** | **6** (`person` 0.91, `bowl` 0.72, `bowl` 0.70, `oven` 0.69, `oven` 0.63, `person` 0.61) |
>
> Ngưỡng càng thấp thì càng nhiều hộp: từ 0.60 xuống 0.20, số prediction tăng từ 6 lên 17, tức là gấp gần 3 lần. Ngưỡng cao thì ít hộp hơn nhưng bỏ sót các vật nhỏ hoặc khó thấy (`cup`, `bowl` nhỏ).

### 2.4 Khi hạ ngưỡng, reviewer phải làm thêm gì?

> Hạ ngưỡng sẽ có nhiều hộp hơn, trong đó có nhiều hộp **sai hoặc trùng**, nên reviewer phải kiểm tra từng hộp (ở ngưỡng 0.20 là 17 hộp thay vì 6):
> - **Xóa hộp thừa**, hộp trùng nhau hoặc hộp nằm trên thứ không phải vật thể.
> - **Sửa nhãn lớp** nếu model đoán sai (ví dụ `bowl` nhưng thực ra là đĩa hoặc nắp).
> - **Chỉnh lại cạnh hộp** cho khớp sát vật.
> - **Kiểm tra vật bị bỏ sót** và thêm hộp còn thiếu.
>
> Khối lượng việc tăng lên nên tốn thêm thời gian và công sức. `score` chỉ là độ tin cậy của model, không phải độ đúng của nhãn, nên người vẫn phải tự quyết định.

### 2.5 Một quy tắc box cho người gán nhãn

> **Quy tắc:** Mỗi vật thể nhìn thấy rõ được vẽ **đúng một hộp riêng**, ôm sát phần **nhìn thấy** của vật, từ mép trái-trên đến mép phải-dưới, không chừa viền thừa và không cắt vào vật. Hai vật cùng lớp thì mỗi vật một hộp riêng. **Không** bỏ vật chỉ vì model score thấp. Quy tắc này áp dụng cho mọi ảnh.

### 2.6 Vật bị che khuất hoặc cắt mép trên ảnh `kitchen`

> **Vật chọn:** `person` score 0.61, `bbox_xyxy = [0.12, 263.18, 61.55, 310.78]`, ở **góc trái dưới** ảnh. Người này **bị cắt bởi mép trái ảnh** và gần như chỉ lộ cánh tay, bàn tay, phần còn lại bị bàn che.
>
> **Quyết định: viết thành quy tắc trong guideline**, vì:
> - Tình huống vật bị cắt mép hoặc che một phần **lặp lại rất nhiều** ở các ảnh khác, nên cần một quy tắc chung để mọi annotator làm giống nhau.
> - Quy tắc đề xuất: *"Nếu vẫn nhận ra được loại vật thì vẽ hộp ôm phần nhìn thấy, cạnh hộp có thể sát mép ảnh. Chỉ **escalation** hỏi Lab Coach khi phần nhìn thấy quá ít để chắc đó là vật gì."*
> - Riêng trường hợp chỉ thấy một bàn tay như ở đây thì ghi chú là ca biên giới để Lab Coach quyết định. Quy tắc chung vẫn nằm trong guideline.

---

## 3. Phân đoạn theo từng đối tượng

### 3.1 Chép một record

> - `instance_id` = **kitchen-001**
> - `class_name` = **person**
> - `score` = **0.899318**
> - `polygon_point_count` = **348**
> - 3–5 điểm đầu của `polygon_xy` = **[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]**

### 3.2 So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

> Box chỉ là một hình chữ nhật (`bbox_xyxy = [385.45, 66.44, 498.02, 348.58]`) nên chỉ cho biết vật nằm trong khoảng nào. Polygon là chuỗi điểm `[x, y]` **bám theo đường biên thật** của vật, nên cho biết thêm:
> - **Hình dạng thật**: người đứng chống tay, có khoảng trống giữa tay và thân, giữa hai chân. Box vẫn tính cả phần nền đó, còn polygon thì không.
> - **Phần nào là vật, phần nào là nền**, nên tính được diện tích thật và tách vật ra khỏi nền.
> - Khi hai vật chồng hoặc chạm nhau, polygon cho thấy ranh giới giữa chúng rõ hơn box.
>
> Đổi lại, polygon có nhiều điểm hơn (348 điểm so với 4 số của box), nên vẽ lâu hơn và dễ sai hơn.

### 3.3 `instance_id` dùng để làm gì? Vì sao không phải class ID và không phải tracking ID?

> - `instance_id` (ví dụ `kitchen-001`) dùng để **phân biệt từng vật thể riêng lẻ** trong output của lab. Hai vật cùng lớp vẫn có `instance_id` khác nhau, ví dụ `kitchen-001` và `kitchen-009` đều là `person` nhưng là hai người, mỗi người một polygon riêng.
> - **Không phải class ID**: class ID (`class_id = 0` cho `person`) là mã của **loại** vật theo taxonomy COCO-80, mọi `person` đều chung một số. `instance_id` thì mỗi vật một mã.
> - **Không phải tracking ID**: tracking ID dùng để theo dõi **cùng một vật qua nhiều khung hình** của video. `instance_id` chỉ là số thứ tự đặt tên trong ảnh này, được sinh theo `sample_id` và thứ tự trong output, không có ý nghĩa theo dõi vật qua ảnh khác.

### 3.4 Một quy tắc vẽ biên mask

> **Quy tắc:** Mỗi vật thể được vẽ **một polygon khép kín**, các điểm đặt **sát đường biên của phần nhìn thấy** của vật, mật độ điểm dày hơn ở chỗ cong hoặc góc, thưa hơn ở đoạn thẳng. Không đưa nền, bóng đổ hay vật che phía trước vào mask. Khi hai vật chạm hoặc chồng nhau thì biên đi theo **đường tiếp xúc**, và mỗi pixel chỉ thuộc về **một** vật (vật ở phía trước).

### 3.5 Một chỗ biên mờ / hai vật chạm nhau / bị che trên ảnh `kitchen`

> **Chỗ chọn:** vùng góc trái của ảnh, nơi mask `oven` (`kitchen-007`, score 0.49) chồng lên bàn bột (`dining table`, `kitchen-005`), `person` (`kitchen-009`, score 0.43, chỉ thấy cánh tay) và các `bowl`. Trên ảnh, các mask này đè lên nhau và ranh giới giữa lò, bàn, tay người không rõ.
>
> **Quyết định:** **dùng cả hai**, tùy mức độ:
> - **Dùng guideline** cho phần có thể quyết định bằng quy tắc chung: khi hai vật chồng nhau thì vật ở phía trước giữ pixel, mask của vật phía sau dừng lại ở mép vật phía trước, không để hai mask chồng lên cùng một vùng. Ca này lặp lại ở nhiều ảnh (bàn có nhiều đồ vật) nên cần một quy tắc để mọi annotator làm giống nhau.
> - **Escalation hỏi Lab Coach** cho phần thật sự không rõ: vùng tối ở mép trái, chưa chắc là bên trong lò hay phần của cánh tay người. Nếu tự đoán thì mỗi annotator sẽ vẽ một kiểu, làm ground truth không nhất quán.

---

## 4. Vòng đời và kiểm tra chất lượng

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | **Một nhãn lớp cho cả ảnh** (đơn vị nhãn = 1 ảnh), gồm `class_id`, `class_name` và `taxonomy_name`. | Ảnh `kitchen` bị gán `gong` (score 0.42) vì ImageNet-1K không có lớp "kitchen". Ảnh `traffic` có nhiều xe buýt, ô tô, người nhưng chỉ ra một lớp `cab` (0.51). Score cao không có nghĩa là nhãn đúng. | Đọc guideline, nhìn cả ảnh rồi chọn **một** lớp theo quy tắc (chủ thể chính hoặc bối cảnh chung) từ danh sách lớp cho trước. Ca khó thì hỏi Lab Coach (escalation), không tự bịa nhãn. | Kiểm tra lớp đã chọn có thuộc taxonomy và đúng với quy tắc guideline không. Soát các ảnh nhiều chủ đề, kiểm tra các annotator có chọn nhất quán không. |
| Phát hiện vật thể | **Một hộp chữ nhật + một lớp cho mỗi vật thể** (đơn vị nhãn = 1 vật), gồm `bbox_xyxy` tính theo pixel. | Ảnh `kitchen`: `person` (0.61) chỉ lộ cánh tay và bị cắt ở mép trái ảnh. `bowl` (0.46) rất nhỏ. `cup` (0.45) chồng lên `bowl` (0.50). `oven` (0.69) có hộp rộng, lẫn cả vùng bàn. | Vẽ **một hộp riêng cho mỗi vật**, ôm sát phần nhìn thấy, chọn đúng lớp. Không bỏ vật chỉ vì score thấp. Vật bị che, cắt mép hoặc quá nhỏ thì làm theo guideline hoặc escalation. | Soát từng hộp: **xóa** hộp thừa hoặc trùng, **sửa** lớp sai, **chỉnh** cạnh hộp cho sát và **thêm** vật bị bỏ sót. Hạ ngưỡng thì số hộp cần soát tăng lên. |
| Instance segmentation | **Một polygon + một lớp + một `instance_id` cho mỗi vật thể** (đơn vị nhãn = 1 vật), gồm danh sách điểm `polygon_xy` bám biên. | Ảnh `kitchen`: mask `oven` (`kitchen-007`, 0.49) chồng lên `dining table` (`kitchen-005`) và `person` (`kitchen-009`), ranh giới không rõ. Polygon `dining table` có tới 598 điểm và có đoạn nhảy lung tung, nên khó coi là biên chuẩn. Mask `spoon` (0.49 và 0.36) rất mảnh. | Vẽ polygon khép kín bám **sát biên phần nhìn thấy** của từng vật. Vật chồng nhau thì mỗi pixel chỉ thuộc một vật (vật phía trước). Đặt `instance_id` riêng cho từng vật. | Kiểm tra polygon có bám biên không, có tự cắt hoặc nhảy điểm không, hai mask có chồng nhau không, lớp và `instance_id` có đúng và duy nhất không. |

> Lưu ý: prediction của model chỉ để quan sát. Ground truth phải do annotator tạo theo guideline và reviewer kiểm tra (QC/rework), không lấy `score` của model làm đáp án.

---

## 5. An toàn dữ liệu

### 5.1 Quy tắc bảo vệ dữ liệu của bạn cho bài lab này (1 câu)

> Chỉ dùng đúng ba ảnh COCO công khai có sẵn trong bài lab, không tải ảnh cá nhân hay dữ liệu nội bộ lên Colab hoặc GitHub, và không đưa thông tin nhạy cảm hay file ZIP vào repo nộp bài.

### 5.2 Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

> Mình sẽ **dừng ngay**, không chạy tiếp, không tải lên hay commit dữ liệu đó, xóa nó khỏi môi trường Colab nếu đã lỡ tải, rồi **báo cho Lab Coach** (và mentor nếu là lỗi kỹ thuật như checksum sai), thay vì tự quyết định hay tự xử lý.
