const person = {
  name: "John",
  age: 30,
  city: "New York"
};

document.getElementById("demo").innerHTML = person.name + ", " + person.age + ", " + person.city;

prodigy.addEventListener('prodigyanswer', event => {
    const selected = event.detail.task.accept || []
    if (!selected.length) {
        alert('Task with no selected options submitted!!')
    }
})
