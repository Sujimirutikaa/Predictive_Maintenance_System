// Additional JavaScript functionality can be added here

// Real-time input validation
document.addEventListener('DOMContentLoaded', function() {
    const inputs = document.querySelectorAll('input[type="number"]');
    
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            validateInput(this);
        });
    });
});

function validateInput(input) {
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    const value = parseFloat(input.value);
    
    // Remove any existing validation classes
    input.classList.remove('is-valid', 'is-invalid');
    
    if (isNaN(value)) {
        return;
    }
    
    if (value < min || value > max) {
        input.classList.add('is-invalid');
    } else {
        input.classList.add('is-valid');
    }
}

// Export prediction results
function exportResults() {
    const results = document.getElementById('predictionResults').innerText;
    const blob = new Blob([results], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'prediction_results.txt';
    a.click();
    window.URL.revokeObjectURL(url);
}

// Print results
function printResults() {
    const printContent = document.getElementById('predictionResults').innerHTML;
    const printWindow = window.open('', '', 'height=600,width=800');
    printWindow.document.write('<html><head><title>Prediction Results</title>');
    printWindow.document.write('<style>body{font-family:Arial,sans-serif;padding:20px;}</style>');
    printWindow.document.write('</head><body>');
    printWindow.document.write('<h1>Predictive Maintenance Results</h1>');
    printWindow.document.write(printContent);
    printWindow.document.write('</body></html>');
    printWindow.document.close();
    printWindow.print();
}