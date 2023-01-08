const canvasElement = document.getElementById('draw-area');
const canvas = new HandwritingCanvas(canvasElement);

const clearButtonElement = document.getElementById('clear-button');
clearButtonElement.addEventListener('click', () => {
    canvas.clear();
});

const predictButton = document.getElementById('predict-button');
predictButton.addEventListener('click', async () => {
    if(canvas.isEmpty) {
        return;
    }

    const blob = await canvas.toBlob('image/png');
    const formData = new FormData();
    formData.append('image', blob, 'number.png');

    const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
    });
    const responseData = await response.json();

    const imageUrl = URL.createObjectURL(blob);
    const imgElement = document.createElement('img');
    imgElement.src = imageUrl;

    const resultImageElement = document.getElementById('result-image');
    if(resultImageElement.firstChild) {
        resultImageElement.removeChild(resultImageElement.firstChild);
    }
    resultImageElement.appendChild(imgElement);
    canvas.clear();


    const tableBodyElement = document.getElementById('result-table-body');
    while(tableBodyElement.firstChild) {
        tableBodyElement.removeChild(tableBodyElement.firstChild);
    }

    const probabilities = responseData.probabilities;
    for(let i = 0; i < probabilities.length; i++) {
        const tr = document.createElement('tr');

        const tdNumber = document.createElement('td');
        tdNumber.textContent = i;
        tr.append(tdNumber);

        const tdProbability = document.createElement('td');
        tdProbability.textContent = (probabilities[i]*100).toFixed(1);
        tr.append(tdProbability);

        tableBodyElement.appendChild(tr);
    }
});
