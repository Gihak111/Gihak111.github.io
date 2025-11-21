
document.addEventListener('DOMContentLoaded', function() {
    // Load Korean locale for Flatpickr
    flatpickr.localize(flatpickr.l10ns.ko);
// DOM Elements
    const taskForm = document.getElementById('task-form');
    const taskTitle = document.getElementById('task-title');
    const taskDescription = document.getElementById('task-description');
const taskList = document.getElementById('task-list');
    const taskCount = document.getElementById('task-count');
    const clearCompletedBtn = document.getElementById('clear-completed');
    const filterButtons = document.querySelectorAll('.filter-btn');
    const dateInput = document.createElement('input');
    dateInput.type = 'text';
    dateInput.className = 'date-input px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500';
    dateInput.placeholder = '마감일 선택';
    taskForm.insertBefore(dateInput, taskForm.lastElementChild);
    
    // Initialize Flatpickr for Korean
    flatpickr(dateInput, {
      locale: "ko",
      dateFormat: "Y-m-d",
      minDate: "today",
      allowInput: true
    });
// State
    let tasks = JSON.parse(localStorage.getItem('tasks')) || [];
    let currentFilter = 'all';
    
    // Initialize
    renderTasks();
    updateTaskCount();
    
    // Event Listeners
    taskForm.addEventListener('submit', addTask);
    clearCompletedBtn.addEventListener('click', clearCompletedTasks);
    filterButtons.forEach(btn => btn.addEventListener('click', filterTasks));
    
    // Functions
    function addTask(e) {
        e.preventDefault();
        const title = taskTitle.value.trim();
        const description = taskDescription.value.trim();
        
        if (title) {
            const newTask = {
                id: Date.now(),
                title: title,
                description: description,
completed: false,
                createdAt: new Date().toISOString(),
                dueDate: dateInput.value || null
            };
            dateInput.value = ''; // Reset date input after adding task
tasks.push(newTask);
            saveTasks();
            renderTasks();
            updateTaskCount();
            taskTitle.value = '';
            taskDescription.value = '';
}
    }
    
    function renderTasks() {
        taskList.innerHTML = '';
        
        if (tasks.length === 0) {
            taskList.innerHTML = '<li class="p-4 text-center text-gray-500">할 일이 없습니다. 위에서 추가해보세요!</li>';
return;
        }
        
        let filteredTasks = tasks;
        
        if (currentFilter === 'active') {
            filteredTasks = tasks.filter(task => !task.completed);
        } else if (currentFilter === 'completed') {
            filteredTasks = tasks.filter(task => task.completed);
        }
        
        if (filteredTasks.length === 0) {
            let message = '';
            if (currentFilter === 'active') {
                message = '진행중인 할 일이 없습니다!';
            } else if (currentFilter === 'completed') {
                message = '완료된 할 일이 없습니다!';
            } else {
                message = '할 일이 없습니다. 위에서 추가해보세요!';
            }
taskList.innerHTML = `<li class="p-4 text-center text-gray-500">${message}</li>`;
            return;
        }
        
        filteredTasks.forEach(task => {
            const taskItem = document.createElement('li');
            taskItem.className = `task-item p-4 flex items-center ${task.completed ? 'completed' : ''}`;
            taskItem.dataset.id = task.id;
            taskItem.draggable = true;
            
            taskItem.innerHTML = `
                <div class="flex items-center flex-1">
                    <input 
                        type="checkbox" 
                        class="mr-3 h-5 w-5 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 cursor-pointer"
                        ${task.completed ? 'checked' : ''}
                    >
                    <div class="flex-1">
                        <div class="task-title font-medium">${task.title}</div>
                        ${task.description ? `<div class="task-description text-sm text-gray-500 mt-1">${task.description}</div>` : ''}
                    </div>
${task.dueDate ? `<span class="due-date ml-2 text-xs px-2 py-1 bg-indigo-100 text-indigo-800 rounded-full">${formatDueDate(task.dueDate)}</span>` : ''}
<input 
                        type="text" 
                        class="edit-input hidden flex-1 px-2 py-1 border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-indigo-500"
                        value="${task.title}"
>
                </div>
                <div class="flex gap-2">
                    <button class="edit-btn p-1 text-gray-500 hover:text-indigo-600">
                        <i data-feather="edit-2" class="w-4 h-4"></i>
                    </button>
                    <button class="delete-btn p-1 text-gray-500 hover:text-red-600">
                        <i data-feather="trash-2" class="w-4 h-4"></i>
                    </button>
                </div>
            `;
            
            taskList.appendChild(taskItem);
            
            // Add event listeners to the new elements
            const checkbox = taskItem.querySelector('input[type="checkbox"]');
            const editBtn = taskItem.querySelector('.edit-btn');
            const deleteBtn = taskItem.querySelector('.delete-btn');
            const taskText = taskItem.querySelector('.task-text');
            const editInput = taskItem.querySelector('.edit-input');
            
            checkbox.addEventListener('change', () => toggleTaskComplete(task.id));
            deleteBtn.addEventListener('click', () => deleteTask(task.id));
            
            editBtn.addEventListener('click', () => {
                taskItem.classList.add('editing');
                taskText.classList.add('hidden');
                editInput.classList.remove('hidden');
                editInput.focus();
            });
            
            editInput.addEventListener('blur', () => saveEdit(task.id, editInput));
            editInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    saveEdit(task.id, editInput);
                }
            });
            
            // Drag and drop events
            taskItem.addEventListener('dragstart', handleDragStart);
            taskItem.addEventListener('dragover', handleDragOver);
            taskItem.addEventListener('drop', handleDrop);
            taskItem.addEventListener('dragend', handleDragEnd);
        });
        
        feather.replace();
    }
    
    function toggleTaskComplete(id) {
        tasks = tasks.map(task => 
            task.id === id ? {...task, completed: !task.completed} : task
        );
        saveTasks();
        renderTasks();
        updateTaskCount();
    }
    
    function deleteTask(id) {
        tasks = tasks.filter(task => task.id !== id);
        saveTasks();
        renderTasks();
        updateTaskCount();
    }
    
    function saveEdit(id, inputElement) {
        const newText = inputElement.value.trim();
        if (newText) {
            tasks = tasks.map(task => 
                task.id === id ? {...task, text: newText} : task
            );
            saveTasks();
            renderTasks();
        } else {
            renderTasks();
        }
    }
    
    function clearCompletedTasks() {
        tasks = tasks.filter(task => !task.completed);
        saveTasks();
        renderTasks();
        updateTaskCount();
    }
    
    function filterTasks(e) {
        currentFilter = e.target.dataset.filter;
        
        filterButtons.forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.filter === currentFilter) {
                btn.classList.add('active');
            }
        });
        
        renderTasks();
    }
    
    function updateTaskCount() {
        const activeTasks = tasks.filter(task => !task.completed).length;
        taskCount.textContent = activeTasks;
    }
    function saveTasks() {
        localStorage.setItem('tasks', JSON.stringify(tasks));
    }

    function formatDueDate(dateString) {
        if (!dateString) return '';
        const date = new Date(dateString);
        const now = new Date();
        const diffTime = date - now;
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        
        if (diffDays === 0) return '오늘 마감';
        if (diffDays === 1) return '내일 마감';
        if (diffDays < 0) return `${Math.abs(diffDays)}일 지남`;
        
        return `${diffDays}일 남음`;
    }

    // Render calendar with tasks
    function renderCalendarTasks() {
        const calendar = document.querySelector('custom-calendar');
        if (calendar) {
            // Implement logic to highlight dates with tasks
            // This would require querying your tasks array for dates
        }
    }
// Drag and Drop Functions
    let draggedItem = null;
    
    function handleDragStart(e) {
        draggedItem = this;
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/html', this.innerHTML);
        this.classList.add('dragging');
    }
    
    function handleDragOver(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
    }
    
    function handleDrop(e) {
        e.stopPropagation();
        e.preventDefault();
        
        if (draggedItem !== this) {
            const draggedId = parseInt(draggedItem.dataset.id);
            const targetId = parseInt(this.dataset.id);
            
            // Find indexes
            const draggedIndex = tasks.findIndex(task => task.id === draggedId);
            const targetIndex = tasks.findIndex(task => task.id === targetId);
            
            // Reorder array
            const [removed] = tasks.splice(draggedIndex, 1);
            tasks.splice(targetIndex, 0, removed);
            
            saveTasks();
            renderTasks();
        }
    }
    
    function handleDragEnd() {
        this.classList.remove('dragging');
    }
});