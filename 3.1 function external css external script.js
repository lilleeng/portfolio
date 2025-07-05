let sum = 0;
let mode = "display";
let state = 'FI';
let expressionArray = [];
let expressionString = '';
// let numbers = '0123456789';
// let operators = ',+-×÷';

function updateDisplay(displayMode) {
    switch (displayMode) {
        case 'INPUTTING':
            document.getElementById('display').innerText = expressionString;
            break;
        case 'RESULT':
            document.getElementById('display').innerText = sum;
            break;
        case 'NULL':
            document.getElementById('display').innerText = 'Ø';
    }
}

function input(n) {
    /*
    Cases:
    -first input                FI
    -operator (+-×÷)            OP
    -integer                    IN
    -decimal                    DE
    -previous input comma       PIC
    */

    switch (n) {
        case 'AC':
            mode = "display";
            state = 'FI';
            expressionArray = [];
            expressionString = '';
            updateDisplay('NULL');
            return;
        case '0':
            switch (state) {
                case 'FI':
                case 'OP':
                    expressionArray.push(Number(n));
                    expressionString += n;
                    state = 'IN';
                    break;
                case 'IN': 
                case 'DE':
                    expressionArray[expressionArray.length-1] = (Number((expressionArray[expressionArray.length-1] + '') + n));
                    expressionString += n;
                    // state remains the same
                    break;
                case 'PIC':
                    expressionArray[expressionArray.length-1] = Number((expressionArray[expressionArray.length-1] + '') + '.' + n);
                    expressionString += n;
                    state = 'DE';
                    break;
            }
            break;
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            switch (state) {
                case 'FI':
                case 'OP':
                    expressionArray.push(Number(n));
                    expressionString += n;
                    state = 'IN';
                    break;
                case 'IN':
                case 'DE':
                    expressionArray[expressionArray.length-1] = (Number((expressionArray[expressionArray.length-1] + '') + n));
                    expressionString += n;
                    // state remains the same
                    break;
                case 'PIC':
                    expressionArray[expressionArray.length-1] = Number((expressionArray[expressionArray.length-1] + '') + '.' + n);
                    expressionString += n;
                    state = 'DE';
                    break;
            }
            break;
        case '+':
        case '-':
        case '×':
        case '÷':
            switch (state) {
                case 'FI':
                case 'OP':
                case 'PIC':
                    // state remains the same
                    break;
                case 'IN':
                case 'DE':
                    expressionArray.push(n);
                    expressionString += n;
                    state = 'OP';
                    break;
            }
            break;
        case ',':
            switch (state) {
                case 'FI':
                case 'OP':
                case 'DE':
                case 'PIC':
                    // state remains the same
                    break;
                case 'IN':
                    expressionString += n;
                    state = 'PIC';
                    break;
            }
            break;
    }
    updateDisplay('INPUTTING');
}

function calculate() {
    for (let i = 0; i < expressionArray.length; i++) {
        if (expressionArray[i] == '') {
            continue;
        }
        if (expressionArray[i] == '÷') {
            //find adjacent terms' indices
            let li = i-1;
            let ri = i+1;
            let j = 1;
            while (expressionArray[i-j] == '') {
                j++;
                li = i-j;
            }
            j = 1;
            while (expressionArray[i+j] == '') {
                j++;
                ri = i+j;
            }
            expressionArray[li] = Number(expressionArray[li]) / Number(expressionArray[ri]);
            expressionArray[i] = '';
            expressionArray[ri] = '';
        }
    }
    for (let i = 0; i < expressionArray.length; i++) {
        if (expressionArray[i] == '') {
            continue;
        }
        if (expressionArray[i] == '×') {
            //find adjacent terms' indices
            let li = i-1;
            let ri = i+1;
            let j = 1;
            while (expressionArray[i-j] == '') {
                j++;
                li = i-j;
            }
            j = 1;
            while (expressionArray[i+j] == '') {
                j++;
                ri = i+j;
            }
            expressionArray[li] = Number(expressionArray[li]) * Number(expressionArray[ri]);
            expressionArray[i] = '';
            expressionArray[ri] = '';
        }
    }
    sum = 0;
    let currentOperation = '+';
    for (let i = 0; i < expressionArray.length; i++) {
        if (expressionArray[i] == '') {
            continue;
        }
        if ('+-'.includes(expressionArray[i])) {
            currentOperation = expressionArray[i];
            continue;
        }
        if (currentOperation == '+') {
            sum += Number(expressionArray[i]);
            continue;
        }
        if (currentOperation == '-') {
            sum -= Number(expressionArray[i]);
            continue;
        }
    }
    updateDisplay('RESULT');
}