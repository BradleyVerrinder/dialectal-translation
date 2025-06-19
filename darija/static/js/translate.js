document.addEventListener('DOMContentLoaded', function() {

    const translationForm = document.getElementById('translationForm');


    if (translationForm){
        translationForm.addEventListener('submit', function(e) {
            e.preventDefault(); 
            
            
            let formData = {
                'input_text': document.getElementById('sourceText').value,
                'model_type': document.getElementById('modelType').value
            };
    
            const inputText = document.getElementById('sourceText').value;
    
            console.log(inputText);
            const csrftoken = document.querySelector('[name=csrfmiddlewaretoken]').value;
            console.log(modelType)
    
            fetch('home', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest',
                    'X-CSRFToken': csrftoken
                },
                body: JSON.stringify(formData)
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('translatedText').value = data.translated_text;
            })
            .catch(error => console.error('Error:', error));
        });
    }
});