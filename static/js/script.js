function showDialog(message) {
    const dialog = document.getElementById('dialog');
    const dialogMessage = document.getElementById('dialog-message');
    dialogMessage.textContent = message;// 設置對話框訊息
    dialog.style.display = 'block';     // 顯示對話框

    const overlay = document.getElementById('overlay');
    overlay.style.display = 'block'; // 顯示遮罩層
}

function closeDialog() {
    const dialog = document.getElementById('dialog');
    dialog.style.display = 'none'; // 隱藏對話框

    const overlay = document.getElementById('overlay');
    overlay.style.display = 'none'; // 隱藏遮罩層
}

function validateFile(event) {
    const allowedExtensions = /(\.jpg|\.jpeg|\.png)$/i; // 允許的檔案擴展名
    const fileInput = document.querySelector('input[type="file"]');
    const file = fileInput.files[0];    // 獲取選中的檔案

    // 檢查檔案是否被選擇
    if (!file) {
        event.preventDefault(); // 阻止表單提交
        showDialog('請選擇一個檔案！'); 
        return false;
    }

    // 檢查檔案名稱
    if (!allowedExtensions.exec(file.name)) {
        event.preventDefault(); // 阻止表單提交
        showDialog('請上傳有效的檔案類型（.jpg, .jpeg, .png）');
        fileInput.value = ''; // 清除檔案輸入
        return false;
    }

    // 檢查檔案大小（限制在 1MB 以內）
    const maxSize = 1024 * 1024;        // 1MB
    if (file.size > maxSize) {
        event.preventDefault(); // 阻止表單提交
        showDialog('檔案大小不能超過 1MB！');
        fileInput.value = '';           // 清除檔案輸入
        return false;
    }

    return true;                        // 檔案有效
}

// 綁定表單提交事件
document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    form.addEventListener('submit', function(event) {
        const isValid = validateFile(event); // 將事件物件傳遞給 validateFile 函數
        if (!isValid) {
            event.preventDefault(); // 如果不有效，阻止表單提交
        }
    });
});