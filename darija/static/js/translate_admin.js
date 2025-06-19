document.addEventListener('DOMContentLoaded', function() {
    const sourceTextArea = document.getElementById('sourceText');
    const translatedTextArea = document.getElementById('translatedText');

    // Event listener for when translation is performed or text areas are updated
    sourceTextArea.addEventListener('input', function() {
        document.querySelector('input[name="source_text"]').value = sourceTextArea.value;
    });

    translatedTextArea.addEventListener('input', function() {
        document.querySelector('input[name="translated_text"]').value = translatedTextArea.value;
    });

    const translationForm = document.getElementById('translationForm');


    if (translationForm){
        translationForm.addEventListener('submit', function(e) {
            e.preventDefault();  // Prevent the default form submission
            
            
            let formData = {
                'input_text': document.getElementById('sourceText').value,
                'model_type': document.getElementById('modelType').value
            };
    
            const inputText = document.getElementById('sourceText').value;
    
            console.log(inputText);
            const csrftoken = document.querySelector('[name=csrfmiddlewaretoken]').value;
            console.log(modelType)
    
            fetch('translate_admin', {
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
                // Update hidden inputs
                document.querySelector('input[name="translated_text"]').value = data.translated_text;
            })
            .catch(error => console.error('Error:', error));
        });
    }
});